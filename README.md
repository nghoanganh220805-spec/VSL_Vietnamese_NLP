# VSL Vietnamese NLP

Module xử lý câu tiếng Việt cho hệ thống nhận diện ngôn ngữ ký hiệu Việt Nam.

## Chức năng

- Lưu từ/cụm từ bằng Sentence Buffer
- Ghép chuỗi ký hiệu thô thành câu tiếng Việt
- Ưu tiên rule/template
- Dùng BARTpho fine-tuned để dự đoán câu
- Fallback sang bộ sửa token đơn giản nếu chưa train BARTpho

## Ví dụ

Input:

```text
toi muon nuoc