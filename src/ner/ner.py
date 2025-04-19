import torch
import torch.nn as nn
import fasttext
from torchcrf import CRF
from typing import Any, Tuple 
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class BiRNN_CRF_FastText(nn.Module):
    def __init__(self, fasttext_model: Any, hidden_dim: int, tagset_size: int):
        super(BiRNN_CRF_FastText, self).__init__()

        self.fasttext_model = fasttext_model
        self.embedding_dim = self.fasttext_model.get_dimension()

        # LSTM layer: Bidirectional LSTM with hidden size divided by 2 (for both directions)
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            bidirectional=True,
            batch_first=True
        )

        # Linear layer to map the LSTM output to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        
        # CRF layer for sequence labeling
        self.crf = CRF(tagset_size)

    def embed_tokens(self, token_lists):
        batch_embeddings = []
        lengths = []

        for tokens in token_lists:
            # Convert each token to its FastText embedding
            vecs = [torch.tensor((tok)).to(device) for tok in tokens]
            batch_embeddings.append(torch.stack(vecs))
            lengths.append(len(tokens))

        # Padding the sequences to the same length
        padded = nn.utils.rnn.pad_sequence(batch_embeddings, batch_first=True, padding_value=0.0)
        lengths = torch.tensor(lengths, dtype=torch.long)

        # Return the padded embeddings and lengths (both on the correct device)
        return padded.to(device), lengths.to(device)

    def forward(self, token_lists):
        # token_lists is a List[List[str]]
        embeddings, lengths = self.embed_tokens(token_lists)



        # Pack the sequences to handle varying lengths
        packed = nn.utils.rnn.pack_padded_sequence(embeddings, lengths.cpu(), batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed)
        
        # Unpack the sequences
        output, _ = nn.utils.rnn.pad_packed_sequence(lstm_out, batch_first=True)

        # Get emissions for CRF
        emissions = self.hidden2tag(output)
        return emissions, lengths

    def loss(self, emissions, tags, lengths):
        # Create a mask to prevent padding tokens from affecting loss
        mask = torch.arange(tags.size(1), device=tags.device)[None, :] < lengths[:, None]
        return -self.crf(emissions, tags, mask=mask)

    def predict(self, emissions, lengths):
        # Create a mask for prediction as well
        mask = torch.arange(emissions.size(1), device=emissions.device)[None, :] < lengths[:, None]
        return self.crf.decode(emissions, mask=mask)
