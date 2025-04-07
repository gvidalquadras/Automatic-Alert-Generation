import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics import classification_report
import pandas as pd
from datasets import load_dataset
import os
from transformers import pipeline
from load_data import add_sentiment_and_save
import ast
from sklearn.metrics import classification_report

def load_data_from_csv(train_file, val_file, test_file):
    train_df = pd.read_csv(train_file)
    val_df = pd.read_csv(val_file)
    test_df = pd.read_csv(test_file)
    
    def process_dataframe(df):
        sentences = []
        tags = []
        sentiments = []
        
        for sentence_id, group in df.groupby('id'):
            sentences.append(group['tokens'].tolist())
            ner_tags_list = group['ner_tags'][sentence_id].replace(' ', ',') 
            ner_tags_list = ast.literal_eval(ner_tags_list)
            tags.append([int(tag) for tag in ner_tags_list])
            sentiments.append(group['sentiment'].iloc[0])
            
        return sentences, tags, sentiments
    
    train_sentences, train_tags, train_sentiments = process_dataframe(train_df)
    val_sentences, val_tags, val_sentiments = process_dataframe(val_df)
    test_sentences, test_tags, test_sentiments = process_dataframe(test_df)
    
    return (train_sentences, train_tags, train_sentiments), (val_sentences, val_tags, val_sentiments), (test_sentences, test_tags, test_sentiments)
# Define CRF (Conditional Random Field) Class
class CRF(nn.Module):
    def __init__(self, hidden_dim, tagset_size):
        super(CRF, self).__init__()
        self.transitions = nn.Parameter(torch.randn(tagset_size, tagset_size))
        self.transitions.data[:, 0] = -10000.0  # No transitions to <PAD>
        self.transitions.data[0, :] = -10000.0  # No transitions from <PAD>
    
    def forward(self, feats, masks):
        return self._forward_alg(feats, masks)
    
    def _forward_alg(self, feats, masks):
        batch_size, seq_len, tagset_size = feats.size()
        alpha = feats[:, 0]
        
        for i in range(1, seq_len):
            emit_scores = feats[:, i].unsqueeze(2)
            trans_scores = self.transitions.unsqueeze(0)
            alpha_expanded = alpha.unsqueeze(1)
            scores = alpha_expanded + trans_scores + emit_scores
            alpha = torch.logsumexp(scores, dim=2)
            mask = masks[:, i].unsqueeze(1)
            alpha = alpha * mask + alpha_expanded.squeeze(1) * (1 - mask)
        
        return alpha
    
    def score_sentence(self, feats, tags, masks):
        batch_size, seq_len, tagset_size = feats.size()
        emit_scores = torch.zeros_like(tags, dtype=torch.float).to(feats.device)
        for i in range(batch_size):
            for j in range(seq_len):
                if masks[i, j]:
                    emit_scores[i, j] = feats[i, j, tags[i, j]]
        
        transitions_score = torch.zeros_like(tags[:, :-1], dtype=torch.float).to(feats.device)
        for i in range(batch_size):
            for j in range(seq_len - 1):
                if masks[i, j + 1]:
                    transitions_score[i, j] = self.transitions[tags[i, j], tags[i, j + 1]]
        
        score = emit_scores.sum(dim=1) + transitions_score.sum(dim=1)
        return score
    
    def viterbi_decode(self, feats, masks):
        batch_size, seq_len, tagset_size = feats.size()
        viterbi_scores = feats[:, 0]
        backpointers = []
        
        for i in range(1, seq_len):
            viterbi_expanded = viterbi_scores.unsqueeze(2)
            emission_expanded = feats[:, i].unsqueeze(1)
            trans_expanded = self.transitions.unsqueeze(0)
            score = viterbi_expanded + trans_expanded + emission_expanded
            viterbi_scores, best_paths = score.max(dim=1)
            mask = masks[:, i].unsqueeze(1)
            viterbi_scores = viterbi_scores * mask + viterbi_expanded.squeeze(2) * (1 - mask)
            backpointers.append(best_paths)
        
        best_tag_seqs = []
        for i in range(batch_size):
            valid_len = masks[i].sum().item()
            if valid_len == 0:
                best_tag_seqs.append([])
                continue
            
            best_last_tag = viterbi_scores[i].argmax().item()
            best_tags = [best_last_tag]
            for bp in reversed(backpointers[:(valid_len-1)]):
                best_last_tag = bp[i][best_tags[-1]].item()
                best_tags.append(best_last_tag)
            
            best_tags.reverse()
            best_tag_seqs.append(best_tags)
        
        return best_tag_seqs

# BiLSTM-CRF Model for Named Entity Recognition
class BiLSTM_CRF(nn.Module):
    def __init__(self, vocab_size, tagset_size, embedding_dim, hidden_dim, sentiment_feature=True, num_layers=1, dropout=0.5):
        super(BiLSTM_CRF, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.use_sentiment = sentiment_feature
        input_dim = embedding_dim
        if sentiment_feature:
            input_dim = embedding_dim + 1
        
        self.lstm = nn.LSTM(input_dim, hidden_dim // 2, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.dropout = nn.Dropout(dropout)
        self.crf = CRF(hidden_dim, tagset_size)
        self.hidden_dim = hidden_dim
        self.tagset_size = tagset_size
        
    def _get_lstm_features(self, sentences, sentiments=None):
        embedded = self.embedding(sentences)
        embedded = self.dropout(embedded)
        
        if self.use_sentiment and sentiments is not None:
            sentiments_expanded = sentiments.unsqueeze(1).unsqueeze(2).expand(-1, embedded.size(1), 1).float()
            embedded = torch.cat([embedded, sentiments_expanded], dim=2)
        
        lstm_out, _ = self.lstm(embedded)
        lstm_out = self.dropout(lstm_out)
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def forward(self, sentences, sentiments=None, tags=None, masks=None):
        if masks is None:
            masks = (sentences != 0).float()
        
        lstm_feats = self._get_lstm_features(sentences, sentiments)
        
        if tags is not None:
            forward_score = self.crf(lstm_feats, masks)
            gold_score = self.crf.score_sentence(lstm_feats, tags, masks)
            loss  = (forward_score.sum(dim=1) - gold_score).mean()
            return loss
        else:
            best_paths = self.crf.viterbi_decode(lstm_feats, masks)
            return best_paths

# Dataset Class to handle NER with Sentiment feature
class NERDataset(Dataset):
    def __init__(self, sentences, tags, word2idx, sentiments=None):
        self.sentences = sentences
        self.tags = tags
        self.word2idx = word2idx
        self.sentiments = sentiments
        
    def __len__(self):
        return len(self.sentences)
    
    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        tag = self.tags[idx]
        sentence_idx = [self.word2idx.get(word, self.word2idx['<UNK>']) for word in sentence]
        tag_idx = tag
        
        if self.sentiments is not None:
            sentiment = self.sentiments[idx]
            return torch.tensor(sentence_idx), torch.tensor(tag_idx), torch.tensor(sentiment)
        else:
            return torch.tensor(sentence_idx), torch.tensor(tag_idx)

# Padding function with sentiment support
def collate_fn(batch):
    if len(batch[0]) == 3:  # With sentiment
        sentences, tags, sentiments = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
        masks = (sentences_padded != 0).float()
        return sentences_padded, tags_padded, torch.tensor(sentiments), masks
    else:  # Without sentiment
        sentences, tags = zip(*batch)
        sentences_padded = pad_sequence(sentences, batch_first=True, padding_value=0)
        tags_padded = pad_sequence(tags, batch_first=True, padding_value=0)
        masks = (sentences_padded != 0).float()
        return sentences_padded, tags_padded, masks

# Data loading from CSV


# Hyperparameters
hyperparameters = {
    'embedding_dim': 100,
    'hidden_dim': 200,
    'dropout': 0.5,
    'num_layers': 1,
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 10,
    'use_sentiment': True
}

# Training and Evaluation
def main():
    train_file = "conll2003_train.csv"
    val_file = "conll2003_validation.csv"
    test_file = "conll2003_test.csv"
    
    use_sentiment = hyperparameters['use_sentiment']
    
    if not (os.path.exists(train_file) and os.path.exists(val_file) and os.path.exists(test_file)):
        add_sentiment_and_save(train_file, val_file, test_file)
    
    (train_sentences, train_tags, train_sentiments), (val_sentences, val_tags, val_sentiments), (test_sentences, test_tags, test_sentiments) = load_data_from_csv(train_file, val_file, test_file)
    
    word2idx = {word: idx for idx, word in enumerate(set(word for sentence in train_sentences for word in sentence))}
    word2idx['<UNK>'] = len(word2idx)
    
    tagset_size = len(set(tag for tag_list in train_tags for tag in tag_list))
    
    # Model initialization
    model = BiLSTM_CRF(len(word2idx), tagset_size, embedding_dim=hyperparameters['embedding_dim'], hidden_dim=hyperparameters['hidden_dim'], sentiment_feature=use_sentiment)
    optimizer = optim.Adam(model.parameters(), lr=hyperparameters['learning_rate'])
    
    # Training loop
    for epoch in range(hyperparameters['epochs']):
        model.train()
        for sentences, tags, sentiments, masks in DataLoader(NERDataset(train_sentences, train_tags, word2idx, train_sentiments), batch_size=hyperparameters['batch_size'], collate_fn=collate_fn):
            optimizer.zero_grad()
            loss = model(sentences, sentiments, tags, masks)
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch + 1}/{hyperparameters["epochs"]}, Loss: {loss.item()}')
        
        # Validation
        model.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for sentences, tags, sentiments, masks in DataLoader(NERDataset(val_sentences, val_tags, word2idx, val_sentiments), batch_size=hyperparameters['batch_size'], collate_fn=collate_fn):
                preds = model(sentences, sentiments, masks=masks)
                val_preds.extend(preds)
                val_true.extend(tags)
        
        # Evaluate validation performance
        print("Validation Classification Report:")
        print(classification_report(val_true, val_preds))

    # Testing
    model.eval()
    test_preds, test_true = [], []
    with torch.no_grad():
        for sentences, tags, sentiments, masks in DataLoader(NERDataset(test_sentences, test_tags, word2idx, test_sentiments), batch_size=hyperparameters['batch_size'], collate_fn=collate_fn):
            preds = model(sentences, sentiments, masks=masks)
            test_preds.extend(preds)
            test_true.extend(tags)

    print("Test Classification Report:")
    print(classification_report(test_true, test_preds))

if __name__ == "__main__":
    main()
