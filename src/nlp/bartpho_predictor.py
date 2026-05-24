from pathlib import Path

import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class BartphoSentencePredictor:
    """
    Dùng model BARTpho đã fine-tune để chuyển chuỗi ký hiệu thô
    thành câu tiếng Việt hoàn chỉnh.
    """

    def __init__(self, model_dir: str = "models/bartpho_sentence"):
        self.model_dir = Path(model_dir)

        if not self.model_dir.exists() or not any(self.model_dir.iterdir()):
            raise FileNotFoundError(
                f"Chưa có model BARTpho tại {self.model_dir}. "
                "Hãy chạy: python -m src.training.train_bartpho"
            )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, raw_sentence: str) -> str:
        raw_sentence = " ".join(raw_sentence.strip().split())

        if not raw_sentence:
            return ""

        input_text = "chuyen_cau_ky_hieu: " + raw_sentence

        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            truncation=True,
            max_length=64
        )

        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_length=64,
                num_beams=4,
                early_stopping=True
            )

        sentence = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

        return sentence.strip()