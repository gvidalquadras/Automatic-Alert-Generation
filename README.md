# Automatic Alert Generation

This repository provides a system for automatically generating alerts from short texts such as tweets. The system uses Named Entity Recognition (NER), sentiment classification, and topic classification to summarize information into short, informative alerts in a subject–verb–object format.

---

## Installation Instructions

### 1. Download FastText Embeddings

To use the NER model, you must download pre-trained FastText word vectors:

```
wget https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
gunzip cc.en.300.bin.gz
```

If wget is not available, run 
```
curl -o cc.en.300.bin.gz https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.en.300.bin.gz
```

### 2. Download spaCy Language Model

```
python -m spacy download en_core_web_sm
```

### 3. Install Python Dependencies
Make sure you are using a virtual environment. Then run:
```
pip install -r requirements.txt
```

## Retraining the Models
To retrain any of the models (e.g., NER or sentiment classifier), run the appropriate training scripts.

```
python -m src.ner.train_NER
```
```
python -m src.sa.train_SA
```

This commands will automatically download the data.

## Generating Alerts from New Input
To generate alerts from custom input:

Open the alert_generation.py file.

Replace the value of the tweet variable with your own sentence, for example:

``tweet = "Your sentence goes here"``

Then run:

```
python -m src.automatic_alert.alert_generation
```

The program will print a generated alert based on the input sentence.

### Output Example

#### Input:
Taylor Swift canceled her concert in Paris.

#### NER Output:
(Taylor, B-PER), (Swift, I-PER), (canceled, O), (her, O), (concert, O), (in, O), (Paris, B-LOC)

#### Sentiment:
Negative

#### Generated Alert:
Bad news about Music: Taylor Swift canceled concert.
