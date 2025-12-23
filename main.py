import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from encoder import Encoder

def train_encoder(epoch=10):
    save_path = "encoder_transformer_ag_news.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    def token_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=256)

    tok = data.map(token_function, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")
    
    train = DataLoader(tok["train"], batch_size=32, shuffle=True, collate_fn=collator)
    test = DataLoader(tok["test"], batch_size=32, shuffle=True, collate_fn=collator)

    model = Encoder(tokenizer.vocab_size, 128, 16, 16, 128, 4, 2, 256, 4).to(device)

    criterion = nn.CrossEntropyLoss()
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    model.train()
    t_loss = 0
    total = 0
    correct = 0
    for i in range(epoch):
        for batch in train:
            optim.zero_grad()
            y = model(batch["input_ids"].to(device), batch["attention_mask"].to(device).long())
            loss = criterion(y, batch["labels"].to(device).long())
            loss.backward()
            optim.step()

            pred = y.argmax(dim=-1)

            t_loss += loss.item() * batch["input_ids"].to(device).size(0)
            correct += (pred == batch["labels"].to(device).long()).sum().item()
            total += batch["input_ids"].to(device).size(0)

        print(f"epoch {i+1} | loss {t_loss/total:.4f} | acc {correct/total:.4f}")


    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
    }, save_path)

    print(f"Saved weights to {save_path}")

    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test:
            y = model(batch["input_ids"].to(device), batch["attention_mask"].to(device).long())
            pred = y.argmax(dim=-1)
            correct += (pred ==  batch["labels"].to(device).long()).sum().item()
            total += batch["input_ids"].to(device).size(0)

    print(f"test acc: {correct/total:.4f}")


def encoder_inference(texts):
    device = "cpu"
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    model = Encoder(tokenizer.vocab_size, 128, 16, 16, 128, 4, 2, 256, 4).to(device)

    checkpoint = torch.load("encoder_transformer_ag_news.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    enc = tokenizer(texts, truncation=True, max_length=256, padding=True, return_tensors="pt")

    with torch.no_grad():
        logits = model(enc["input_ids"].to(device), enc["attention_mask"].to(device).long())
        preds = logits.argmax(dim=-1)

    targets = ["World", "Sports", "Business", "Sci/Tech"]
    
    print("Prediction: ", targets[preds])
    probs = nn.functional.softmax(logits)
    print(probs)

if __name__ == "__main__":
    train_encoder()
    # encoder_inference("")