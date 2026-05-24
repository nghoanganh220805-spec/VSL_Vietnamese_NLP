import json
from pathlib import Path

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


CONFIG_PATH = Path("configs/nlp_config.json")


class SentencePairDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_input_len=64, max_target_len=64):
        self.df = dataframe.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.max_input_len = max_input_len
        self.max_target_len = max_target_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        raw_text = str(self.df.loc[idx, "raw_sentence"]).strip()
        target_text = str(self.df.loc[idx, "complete_sentence"]).strip()

        input_text = "chuyen_cau_ky_hieu: " + raw_text

        inputs = self.tokenizer(
            input_text,
            max_length=self.max_input_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        targets = self.tokenizer(
            target_text,
            max_length=self.max_target_len,
            truncation=True,
            padding="max_length",
            return_tensors="pt"
        )

        labels = targets["input_ids"].squeeze(0)
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "labels": labels
        }


def load_config():
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file cấu hình: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def main():
    config = load_config()

    data_path = Path("data/sentence_pairs.csv")
    output_dir = Path(config["trained_model_dir"])

    if not data_path.exists():
        raise FileNotFoundError(f"Không tìm thấy dataset: {data_path}")

    df = pd.read_csv(data_path)
    df = df.dropna()
    df["raw_sentence"] = df["raw_sentence"].astype(str).str.strip()
    df["complete_sentence"] = df["complete_sentence"].astype(str).str.strip()

    print("Số mẫu train:", len(df))
    print("Model nền:", config["model_name"])

    tokenizer = AutoTokenizer.from_pretrained(config["model_name"])
    model = AutoModelForSeq2SeqLM.from_pretrained(config["model_name"])

    dataset = SentencePairDataset(
        dataframe=df,
        tokenizer=tokenizer,
        max_input_len=config["max_input_length"],
        max_target_len=config["max_target_length"]
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config["batch_size"],
        shuffle=True
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Thiết bị train:", device)

    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"]
    )

    model.train()

    for epoch in range(config["num_epochs"]):
        total_loss = 0.0

        for batch in dataloader:
            batch = {key: value.to(device) for key, value in batch.items()}

            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{config['num_epochs']} | Loss: {avg_loss:.4f}")

    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("Đã fine-tune BARTpho xong.")
    print("Model lưu tại:", output_dir)


if __name__ == "__main__":
    main()