# imports

dataset = SADataset(texts, labels, fasttext_model)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

model = SentimentClassifier(embedding_dim=300, hidden_dim=128)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(5):
    for inputs, targets in dataloader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
    
    print(f"✅ Epoch {epoch+1}, Loss: {loss.item():.4f}")
