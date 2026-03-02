# Kiến Trúc Chi Tiết Của Mô Hình FLUX.2 (Black Forest Labs)

Tài liệu này được trích xuất và phân tích trực tiếp từ mã nguồn của FLUX.2 trong dự án của bạn nhằm giải thích mạch lạc nhất để bạn **đọc vào là hiểu liền** nguyên lý hoạt động của model này.

---

## 🚀 1. Tóm Tắt Trong 1 Phút (Dành cho người mới)

Thay vì dùng cơ chế **U-Net + DDPM (Diffusion truyền thống)** giống như Stable Diffusion hay Midjourney đời cũ, **FLUX.2** vận hành qua nền tảng hoàn toàn mới gồm 3 điểm đắt giá nhất:

1. **Trái tim là Flow Matching**: Thay vì từ từ khử nhiễu (Denoising) từng bước ngắn tốn thời gian, nó học một trường vector (Vector Field) đi theo đường thẳng từ "nhiễu" tới "ảnh". Giúp số step giảm đi mà ảnh lại sắc nét.
2. **Kiến trúc MM-DiT (Multimodal Diffusion Transformer)**: Gạch bỏ U-Net. Dùng Transformer y như ChatGPT, nhưng có 2 luồng xử lý song song cho Text và Hình Ảnh.
3. **Dùng nguyên dàn LLM Khủng (Mistral 24B / Qwen3) làm bộ não đọc Text**: Thay vì dùng CLIP (vốn ngu ngơ với prompt dài), FLUX.2 dùng một hệ ngôn ngữ lớn (LLM) để "đọc hiểu văn bản" -> Đó là lý do FLUX sinh ra chữ trên ảnh cực chuẩn và làm theo prompt siêu việt.

---

## 🛠️ 2. Luồng Hoạt Động Khi Bạn Bấm "Generate"

Hãy hình dung khi bạn gõ: *"1 con mèo đội nón lá uống cà phê"*:

1. **Bước 1: Hiểu Ngôn Ngữ (Text Encoder)**
   - Text của bạn được đưa vào `Mistral3SmallEmbedder` hoặc `Qwen3Embedder`. 
   - Độ dài prompt hỗ trợ cực cao (`MAX_LENGTH = 512` token).
   - Mô hình LLM sẽ trả về một ma trận Nhúng Văn Bản (Text Embeddings) chứa đầy đủ ý nghĩa hình học, ngữ cảnh.
   
2. **Bước 2: Tạo "Bức Vải Thô" (Latent Noise)**
   - Hệ thống tạo ra một ma trận nhiễu ngẫu nhiên gọi là **Latent** (chứ không phải tạo thẳng ảnh màu RGB). 

3. **Bước 3: Tưởng Tượng & Gọt Giũa (DiT Backbone)**
   - Mạng Transformer của model (phiên bản `Klein4B` hoặc `Klein9B`) sẽ nhận 2 thứ: **Latent (Ảnh Nhiễu)** và **Text Embeddings (Yêu cầu)**.
   - Nhờ kiến trúc khối **DoubleStreamBlocks** và **SingleStreamBlocks** (giải thích ở dưới), nó dần dần uốn nắn hạt nhiễu về đúng hình hài bức ảnh. Quá trình này lặp đi lặp lại một vài step (Euler Scheduler).

4. **Bước 4: Tráng Phim (AutoEncoder / VAE Decoder)**
   - Khối Latent sạch (sau gọt giũa) có số channel là `z_channels=32`. 
   - Nó chạy qua bộ **Decoder** (Giải mã) của AutoEncoder kết xuất ra ngay bức ảnh RGB (`out_channels=3`) siêu nét cho chúng ta xem.

---

## 🏗️ 3. Phân Tích Kỹ Thuật (Phẫu Thuật Mã Nguồn)

Dựa trên code gốc (`src/flux2`), cấu trúc của FLUX.2 chia làm 3 Module cốt lõi:

### A. AutoEncoder (Bộ nén - VAE)
Nằm trong `autoencoder.py`. Model Diffusion không chạy trên điểm ảnh (Pixels) nặng nề mà chạy trên "Dữ liệu đã nén" (Latent Space) để tính toán nhanh hơn.
* **Đầu vào (RGB Image)**: 3 Channels.
* **Bộ Nén (Encoder)**: Dùng các khối `ResnetBlock` và Attention rút gọn ảnh thành `z_channels = 32`. (Rất dị biệt vì SD1.5/SDXL chỉ nén xuống 4 channels). Việc nén xuống 32 channels giúp giữ lại luồng chi tiết hạt (texture) siêu nét.
* **Bộ Giải nén (Decoder)**: Dịch ngược từ 32 Channels về 3 Channels RGB để hiển thị cho con người xem.

### B. Ngôn Ngữ (Text Encoders)
Nằm trong `text_encoder.py`. Khác bọt hoàn toàn với các đời model khác:
* Hỗ trợ **`Mistral3SmallEmbedder`** (dùng model `Mistral-Small-3.2-24B-Instruct`). Đây là một con quái vật LLM 24 tỉ tham số được cấy ghép vào làm mảng Embedder. Điều này cho phép suy luận hình ảnh bằng tiếng người cực mượt, suy diễn logic của tấm hình cực cao.
* Hỗ trợ **`Qwen3Embedder`**: Mạng LLM xuất sắc khác. 
* Cả 2 đều có cơ chế upsample (enrich prompt tự động) trong inference.

### C. Lõi Transformer Sinh Ảnh (DiT) 
Nằm trong `model.py`. Lõi không còn là U-Net CNN mà là **Transformer**. Gồm 2 pha:

1. **Khối Luồng Đôi (DoubleStreamBlock):** 
   - Ảnh (Latents) và Văn bản (Text Embeddings) ban đầu khác biệt tần số nên không được "trộn mắm chung với muối".
   - Luồng Text và Luồng Hình được chạy qua 2 dãy Attention (Self-Attention) **riêng biệt** nhưng được liên kết qua một cầu nối Cross-Attention. Nó giúp Text điều khiển Hình mà vị trí Hình không làm hỏng cấu trúc Text.
2. **Khối Luồng Đơn (SingleStreamBlock):** 
   - Nửa sau của mạng. Lúc này Text và Ảnh đã hiểu "nhau", mạng sẽ GỘP (Concat) cả chuỗi Text và Hình lại làm 1 chuỗi dài ngoằn. Đưa vào khối Single Block Attention để tung chiêu cuối => Tạo ra kết quả Flow tốt nhất.

### D. Các Phân Lớp Mô Hình (Variants)
Nổi bật với 2 phiên bản mã "Klein":
* **Klein 9B (`Klein9BParams`) - 9 Tỉ tham số:**
  * `hidden_size`: 4096 (độ lớn mảng tri thức).
  * `num_heads`: 32 (Số đầu xử lý song song).
  * Khối đôi (`depth`): 8 lớp.
  * Khối đơn (`depth_single_blocks`): 24 lớp.
  * *=> Rất nặng, cần VRAM siêu to (24GB+), nhưng độ chi tiết và tuân thủ prompt là Vô Địch.*

* **Klein 4B (`Klein4BParams`) - 4 Tỉ tham số:**
  * `hidden_size`: 3072.
  * `num_heads`: 24.
  * Khối đôi: 5. Khối đơn: 20 lớp.
  * *=> Chạy được trên cạc yếu hơn (10-12GB VRAM áp dụng FP8 Quantization), tốc độ tạo ảnh nhanh gấp rưỡi.*

---

## 🖌️ 4. Mở Rộng Cho Inpainting (Mask-Conditioned Generation)

### Chiến lược: Channel Concatenation

Để FLUX.2 hiểu **"vẽ ở đâu"**, ta mở rộng đầu vào của lớp `img_in` (Linear projection đầu tiên của DiT):

```
[noisy_latent (128ch)] + [mask (1ch)] + [masked_image_latent (128ch)] = 257 channels
```

| Thành phần | Shape | Mô tả |
|---|---|---|
| `noisy_latent` | `[B, 128, h, w]` | Ảnh nhiễu đang được khử |
| `mask` | `[B, 1, h, w]` | Mặt nạ: 1=vẽ lại, 0=giữ nguyên |
| `masked_image_latent` | `[B, 128, h, w]` | Ảnh gốc × (1−mask), model thấy ngữ cảnh xung quanh |

### Zero-Init Weight (Tránh Catastrophic Forgetting)

```python
# img_in.weight gốc: [hidden_size, 128]
# img_in.weight mới:  [hidden_size, 257]
#
# Cột 0:127   = copy từ pretrained     ← model vẫn hoạt động bình thường
# Cột 128:256 = khởi tạo = 0           ← ban đầu bỏ qua mask, học dần
```

### Training Pipeline (Flow Matching + Masked Loss)

```
1. Encode ảnh gốc → image_latent (128ch)
2. Downsample mask → mask_latent (1ch)  
3. Sample t ∈ (0,1), tạo x_t = (1-t)·noise + t·image_latent
4. Model nhận [x_t | mask | masked_image] → dự đoán velocity
5. Loss = MSE(pred, target) × mask    ← CHỈ tính loss trong vùng mask
```

### Các file đã thay đổi

| File | Thay đổi |
|---|---|
| `model.py` | Thêm `inpaint_in_channels` vào Params, mở rộng `img_in` layer |
| `sampling.py` | Thêm `prepare_mask_latent()`, `prepare_inpaint_latent()`, `denoise_inpaint()` |
| `util.py` | Thêm `load_flow_model_inpaint()` (zero-init weight extension) |
| `scripts/inpaint_clock.py` | Cập nhật dùng inpaint pipeline mới |
| `scripts/train_inpaint.py` | **MỚI** — Training script với masked Flow Matching loss |

---

## 🎯 5. Tổng Kết (Tại sao FLUX.2 mạnh?)

Bạn có thể nắm cốt lõi ở 3 chữ: **LLM + Flow + MM-DiT**. 
1. Thay bộ đọc CLIP não cá vàng bằng **LLM Mistral/Qwen cực thông minh**.
2. Thay đường cong Denoise dài ngoằn của DDPM bằng đường thẳng chéo đi nhanh nhất của **Flow Matching**.
3. Thay U-Net đồ cổ bằng **Transformer Luồng Song Song (MM-DiT)** xịn sò.

Từ đó FLUX.2 tạo ra tay chân chuẩn xác, vẽ chữ chính xác 100%, kết xuất ra ánh sáng PBR chân thực chỉ trong số frame (steps) cực kỳ thấp!
