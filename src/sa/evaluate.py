import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import fasttext
from src.sa.SA import SentimentClassifier, SADataset
from src.sa.train_SA import collate_fn, load_sentiment140_csv, evaluate
from src.utils import set_seed


def main():
    """
    Main function for evaluating the sentiment analysis model on test data.
    """
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load FastText embedding model
    ft_model = fasttext.load_model("cc.en.300.bin")

    # Load test data
    test_texts, test_labels = load_sentiment140_csv("data/sentiment140_test.csv")
    test_dataset = SADataset(test_texts, test_labels, ft_model)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

    # Initialize model and load weights
    model = SentimentClassifier()
    model.load_state_dict(torch.load("models/SA_model.pth", map_location=device))
    model.to(device)
    model.eval()

    # Define loss function
    criterion = nn.BCEWithLogitsLoss()

    # Evaluate model
    print("Evaluating test data...")
    test_loss, test_acc = evaluate(model, test_loader, criterion, device)
    print(f"\nTest Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()
