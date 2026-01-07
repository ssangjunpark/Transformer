import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorWithPadding
from encoder import Encoder
from decoder import Decoder
from transformer import Transformer

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

def train_decoder(epoch=30):
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
        d_embed=256,
        d_qk=32,
        d_v=32,
        d_model=256,
        n_att=16,
        n_transformers=8,
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
        d_embed=256,
        d_qk=32,
        d_v=32,
        d_model=256,
        n_att=16,
        n_transformers=8,
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


def train_transformer(epoch=5):
    save_path = "transformer_translation_en_ko.pt"
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("Loading dataset...")
    ds = load_dataset("lemon-mint/korean_english_parallel_wiki_augmented_v1")
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    
    seq_len = 128
    batch_size = 32

    def preprocess_function(examples):
        inputs = examples["english"]
        targets = examples["korean"]
        
        model_inputs = tokenizer(inputs, max_length=seq_len, truncation=True)
        labels = tokenizer(targets, max_length=seq_len, truncation=True)
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    print("Tokenizing dataset...")
    tokenized_datasets = ds.map(
        preprocess_function, 
        batched=True, 
        remove_columns=ds["train"].column_names
    )

    def collate_fn(batch):
        src_ids = [item["input_ids"] for item in batch]
        tgt_ids = [item["labels"] for item in batch]
        
        src_batch = tokenizer.pad({"input_ids": src_ids}, padding=True, return_tensors="pt")
        tgt_batch = tokenizer.pad({"input_ids": tgt_ids}, padding=True, return_tensors="pt")
        
        return {
            "src": src_batch["input_ids"],
            "src_mask": src_batch["attention_mask"],
            "tgt": tgt_batch["input_ids"],
            "tgt_mask": tgt_batch["attention_mask"]
        }

    train_loader = DataLoader(tokenized_datasets["train"], batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
    
    model = Transformer(
        num_emb=tokenizer.vocab_size,
        d_model=256,
        d_qk=32,
        d_v=32,
        n_att=8,
        n_enc_blocks=4,
        n_dec_blocks=4,
        max_len=seq_len
    ).to(device)
    
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    
    print("Start Training...")
    for i in range(epoch):
        model.train()
        total_loss = 0
        total_counts = 0
        
        for batch in train_loader:
            optim.zero_grad()
            
            src = batch["src"].to(device)
            src_mask = batch["src_mask"].to(device)
            tgt = batch["tgt"].to(device)
            tgt_mask = batch["tgt_mask"].to(device)
            
            dec_input = tgt[:, :-1]
            dec_target = tgt[:, 1:]
            dec_mask = tgt_mask[:, :-1]
            
            logits = model(src, dec_input, src_mask=src_mask, tgt_mask=dec_mask)
            
            loss = criterion(logits.reshape(-1, tokenizer.vocab_size), dec_target.reshape(-1))
            loss.backward()
            optim.step()
            
            total_loss += loss.item()
            total_counts += 1
            
        print(f"Epoch {i+1} | Loss {total_loss/total_counts:.4f}")
        
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

def inference_transformer(text):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")
    seq_len = 128
    
    model = Transformer(
        num_emb=tokenizer.vocab_size,
        d_model=256,
        d_qk=32,
        d_v=32,
        n_att=8,
        n_enc_blocks=4,
        n_dec_blocks=4,
        max_len=seq_len
    ).to(device)
    
    model.load_state_dict(torch.load("transformer_translation_en_ko.pt", map_location=device))
    model.eval()
    
    src = tokenizer(text, return_tensors="pt", truncation=True, max_length=seq_len).to(device)
    src_ids = src["input_ids"]
    src_mask = src["attention_mask"]
    
    enc_out = model.encode(src_ids, src_mask)
    
    tgt_ids = torch.tensor([[tokenizer.cls_token_id]], device=device)
    
    for _ in range(seq_len):
        tgt_mask = torch.ones_like(tgt_ids)
        out = model.decode(tgt_ids, enc_out, tgt_mask=tgt_mask, enc_mask=src_mask)
        
        next_token_logits = out[:, -1, :]
        next_token = next_token_logits.argmax(dim=-1).unsqueeze(0)
        
        tgt_ids = torch.cat([tgt_ids, next_token], dim=1)
        
        if next_token.item() == tokenizer.sep_token_id:
            break
            
    print(tokenizer.decode(tgt_ids[0], skip_special_tokens=True))


if __name__ == "__main__":
    # train_encoder()
    # encoder_inference("umpire is going to be replaced with sensors")
    # train_decoder()
    # decoder_inference("Hello world! This is")
    # train_transformer(epoch=1) # Train for 1 epoch for demonstration
    # inference_transformer("Hello world")
    train_transformer(epoch=5)
    pass
