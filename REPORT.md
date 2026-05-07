# Lab 21 — Evaluation Report

**Học viên**: Lã Thị Linh — 2A202600089  
**Ngày nộp**: 08/05/2026  
**Submission option**: A (lightweight)

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit`
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up to power of 2, capped at 1024)
- **GPU**: Tesla T4, 16 GB VRAM
- **Training cost**: $0.07 (~12.2 phút @ $0.35/hr)

## 2. Rank Experiment Results

| Rank | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-----------------|------------|-----------|-----------|------------|
| 8    | 1,843,200       | 3.99 min   | 7.21 GB   | 1.5577    | 4.75       |
| 16   | 3,686,400       | 4.25 min   | 6.61 GB   | 1.5160    | 4.55       |
| 64   | 14,745,600      | 3.99 min   | 7.99 GB   | 1.4768    | 4.37       |

## 3. Loss Curve Analysis
[Đính kèm loss_curve.png]
- **Quan sát**: Dựa trên log training của 3 mô hình, Training Loss giảm đều đặn qua các step (ví dụ r=64 giảm từ 1.60 xuống 1.27, r=16 giảm từ 1.61 xuống 1.39). Eval Loss ở cuối mỗi quá trình train (dao động 1.47 - 1.55) không bị cao vọt lên so với Train Loss, đồng thời Perplexity duy trì ở mức khá tốt (4.3 - 4.7). Do notebook chạy trên Tesla T4 đã giới hạn không bật `eval_strategy` trong quá trình train để tiết kiệm VRAM nên chỉ theo dõi được Train Loss Curve, tuy nhiên tổng thể cho thấy mô hình học tốt và không có hiện tượng overfitting rõ rệt sau 3 epochs.

## 4. Qualitative Comparison (5 examples)

### Example 1
**Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động...
- **Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI...
- **Nhận xét**: Cả hai mô hình đều đưa ra định nghĩa chính xác, tuy nhiên mô hình fine-tuned giải thích ngắn gọn, dễ hiểu và đi vào trọng tâm hơn ("không có sự hướng dẫn trực tiếp").

### Example 2
**Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python... `def fibonacci(n): if n <= 0: return "N p...`
- **Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau: `def fibonacci(n): if n < 0: raise ValueError("Input phải là một số nguyên dương.")...`
- **Nhận xét**: Mô hình fine-tuned thể hiện sự cẩn thận hơn khi thêm lệnh `raise ValueError` cho các input sai, cải thiện đáng kể về mặt lập trình.

### Example 3
**Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng...
- **Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị... 3. Đơn giản...
- **Nhận xét**: Output của mô hình fine-tuned đi thẳng vào các keyword cụ thể và rõ ràng hơn (Chuyển đổi, Thích ứng, Đơn giản) thay vì mô tả chung chung như base model.

### Example 4
**Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp...
- **Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả...
- **Nhận xét**: Cả hai mô hình đều đưa ra một số khái niệm có phần "hallucination" (ví dụ FT model giải thích sai chữ viết tắt của LoRA). Điều này cho thấy dataset (Vietnamese Alpaca) có thể không chứa nhiều kiến thức chuyên sâu về LoRA/QLoRA để dạy model.

### Example 5
**Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học...
- **Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa...
- **Nhận xét**: Cả hai model đều nắm được khái niệm cơ bản. Fine-tuned model đưa ra câu trả lời có cấu trúc tiếng Việt tự nhiên và mượt mà hơn.

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất trên dataset này?**: `r=16` mang lại ROI (Return on Investment) tốt nhất. So với `r=8`, nó giảm được Perplexity từ 4.75 xuống 4.55 (cải thiện đáng kể) trong khi chỉ tốn thêm một khoảng thời gian train không đáng kể (tăng khoảng 15 giây) và peak VRAM lại tối ưu hơn do các biến động hệ thống (6.61 GB so với 7.21 GB).
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**: Việc tăng rank lên `r=64` giúp giảm thêm Perplexity xuống 4.37, nhưng đánh đổi lại số lượng trainable parameters tăng vọt lên gấp 4 lần (từ ~3.6 triệu lên ~14.7 triệu), làm tăng rủi ro overfitting và tiêu tốn đỉnh VRAM cao hơn (gần 8GB). Sự cải thiện từ `r=16` lên `r=64` không tương xứng với sự gia tăng về số lượng tham số và tài nguyên tính toán.
- **Recommendation**: Nếu deploy production cho một task có độ phức tạp trung bình (như general instruction following), em sẽ chọn `r=16`. Rank này giữ được sự cân bằng hoàn hảo giữa hiệu suất model (Perplexity thấp), chi phí triển khai (adapter nhỏ, tốn ít VRAM) và thời gian huấn luyện.

## 6. What I Learned
- Quá trình tuning tham số `rank` (r) và `alpha` rất quan trọng; việc tăng `rank` không phải lúc nào cũng tốt mà cần đánh đổi với tài nguyên VRAM và nguy cơ overfitting.
- Việc giới hạn `max_seq_length` (ví dụ dùng p95 của dataset = 562 thay vì max) và sử dụng `gradient checkpointing` là những kỹ thuật cực kì hữu ích để tối ưu VRAM khi train model trên những GPU có giới hạn bộ nhớ như Tesla T4.
- Đánh giá chất lượng mô hình cần kết hợp cả định lượng (Perplexity) và định tính (thử nghiệm với các test prompt), vì đôi khi số liệu tốt nhưng mô hình lại mắc lỗi "hallucination" trên những prompt đòi hỏi kiến thức nằm ngoài dữ liệu được fine-tune.
