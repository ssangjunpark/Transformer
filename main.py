import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from encoder import Encoder
from decoder import Decoder

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
    probs = nn.functional.softmax(logits, dim=-1)[0][preds]
    print(probs)

def train_decoder(epoch=10):
    save_path = "decoder_transformer_ag_news.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    data = load_dataset("ag_news")
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    seq_len = 256
    bs = 32

    def token_function(batch):
        return tokenizer(batch["text"], truncation=True, max_length=seq_len)

    tok = data.map(token_function, batched=True, remove_columns=["text"])

    collator = DataCollatorWithPadding(tokenizer=tokenizer, return_tensors="pt")

    train = DataLoader(tok["train"], batch_size=bs, shuffle=True, collate_fn=collator)
    test  = DataLoader(tok["test"],  batch_size=bs, shuffle=False, collate_fn=collator)

    vocab_size = tokenizer.vocab_size
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0

    model = Decoder(
        num_embed=vocab_size,
        d_embed=128,
        d_qk=16,
        d_v=16,
        d_model=128,
        n_att=4,
        n_transformers=2,
        max_length=seq_len,
        num_vocab=vocab_size,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    optim = torch.optim.AdamW(model.parameters(), lr=3e-4)

    for i in range(epoch):
        model.train()
        t_loss = 0.0
        total_tokens = 0

        for batch in train:
            optim.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            x = input_ids[:, :-1]
            y = input_ids[:, 1:].clone()

            pad_mask = attn_mask[:, :-1]

            y[input_ids[:, 1:] == pad_id] = -100

            logits = model(x, pad_mask=pad_mask)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            loss.backward()
            optim.step()

            n_toks = (y != -100).sum().item()
            t_loss += loss.item() * n_toks
            total_tokens += n_toks

        print(f"epoch {i+1} | loss {t_loss/max(total_tokens,1):.4f}")

    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": vocab_size,
        "tokenizer_name": "distilbert-base-uncased",
        "seq_len": seq_len,
    }, save_path)

    print(f"Saved weights to {save_path}")

    model.eval()
    t_loss = 0.0
    total_tokens = 0
    with torch.no_grad():
        for batch in test:
            input_ids = batch["input_ids"].to(device)
            attn_mask = batch["attention_mask"].to(device)

            x = input_ids[:, :-1]
            y = input_ids[:, 1:].clone()
            pad_mask = attn_mask[:, :-1]

            y[input_ids[:, 1:] == pad_id] = -100

            logits = model(x, pad_mask=pad_mask)
            loss = criterion(logits.reshape(-1, vocab_size), y.reshape(-1))

            n_toks = (y != -100).sum().item()
            t_loss += loss.item() * n_toks
            total_tokens += n_toks

    print(f"test loss: {t_loss/max(total_tokens,1):.4f}")

def decoder_inference(
    text,
    max_new_tokens=50,
    seq_len=256,
    temperature=1.0,
    top_k=50,
):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    tok = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "[PAD]"})
    pad_id = tok.pad_token_id

    ckpt = torch.load("decoder_transformer_ag_news.pt", map_location="cpu")
    vocab_size = ckpt.get("vocab_size", tok.vocab_size)

    model = Decoder(
        num_embed=vocab_size,
        d_embed=128,
        d_qk=16,
        d_v=16,
        d_model=128,
        n_att=4,
        n_transformers=2,
        max_length=seq_len,
        num_vocab=vocab_size,
    ).to(device)

    model.load_state_dict(ckpt["model_state_dict"], strict=True)
    model.eval()

    ids = tok(text, add_special_tokens=False)["input_ids"]
    if len(ids) == 0:
        ids = [tok.cls_token_id] if tok.cls_token_id is not None else [pad_id]

    input_ids = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if input_ids.size(1) > seq_len:
                input_ids = input_ids[:, -seq_len:]

            attn_mask = torch.ones_like(input_ids, dtype=torch.long, device=device)

            x = input_ids
            pad_mask = attn_mask

            logits = model(x, pad_mask=pad_mask)
            next_logits = logits[:, -1, :]

            if temperature is not None and temperature > 0:
                next_logits = next_logits / temperature

            if top_k is not None and top_k > 0:
                k = min(top_k, next_logits.size(-1))
                vals, idx = torch.topk(next_logits, k, dim=-1)
                probs = torch.nn.functional.softmax(vals, dim=-1)
                next_token = idx.gather(-1, torch.multinomial(probs, num_samples=1))
            else:
                probs = torch.nn.functional.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            input_ids = torch.cat([input_ids, next_token], dim=1)

    out = tok.decode(input_ids[0].tolist(), skip_special_tokens=True)
    print(out)
    return out

if __name__ == "__main__":
    # train_encoder()
    # encoder_inference("umpire is going to be replaced with sensors")
    # train_decoder()
    decoder_inference("Hello world! This is")