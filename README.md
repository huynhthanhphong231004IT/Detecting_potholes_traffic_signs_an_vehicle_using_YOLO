<h2 align="center">
  Author: Huynh Thanh Phong (ReoRioll)
</h2>

<p align="center">
   Computer Science of College of Information and Communication Technology of Can Tho University (Course 48)<br>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Researchs:</b> Artificial Intelligence in Education - Mathematics in Deep Learning and Machine Learning<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Name Project:</b></b> </mark> Detecting potholes, traffic signs, and vehicles using YOLO for autonomous vehicles.<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<mark><b><b>Link Data:</b></b> </mark> https://www.kaggle.com/datasets/reorioll/autonomous-vehicle<br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
<b>Timeline:</b> 09/2025 – 10/2026 at AI-IOT Department - Onyx United Foundation
</p>
<p align="center">
   <b>Presional link Information</b>
</p>

<p>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Facbook: https://www.facebook.com/huynh.thanh.phong.561667 <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Kaggle: https://www.kaggle.com/reorioll <br>

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;
Youtobe: https://www.youtube.com/@ReoRioll-2304CICTCTU <br>
</p>
<br>

## Kết quả huấn luyện mô hình Yolo11n trên tập dữ liệu autonomous-vehicle
<p align="center">
  <img src="Charts/correlation_matrix.png" width="800">
  <br>
  <i>Correlation matrix</i>
</p>

<p align="center">
  <img src="Charts/distribution_lr.png" width="900">
  <br>
  <i>Distribution with learning rate</i>
</p>

<p align="center">
  <img src="Charts/distribution_mAP-time_loss.png" width="900">
  <br>
  <i>Distribution mAP, time and loss</i>
</p>

<p align="center">
  <img src="Charts/compare_train_val_loss.png" width="800">
  <br>
  <i>Compare train and val loss</i>
</p>

<p align="center">
  <img src="Charts/compare_train_val_accuracy.png" width="800">
  <br>
  <i>Compare train and val accuracy</i>
</p>

## Biểu thức tổng quan về quy trình xử lý và dựng khung hình của hệ thống
> $$ \mathbf{\textcolor{red}{Frame}}_{\text{out}}^{(t)} = \Psi_{\text{\textcolor{red}{Render}}} \Big( I^{(t)}, \; \mathcal{D}\left(I^{(t)}\right), \; d(H, h, f), \; \mathcal{C}_{\text{\textcolor{red}{ocr}}}^{(t)}\left(g_{\text{\textcolor{red}{OCR}}}\left(I_{\text{crop}}^{(t)}\right)\right), \; S_{\text{\textcolor{red}{light}}}^{(t)}\left(\text{\textcolor{red}{TrafficLightColor}}\left(I^{(t)}, \mathbf{b}^*\right)\right) \Big) $$


1. Điều kiện chạy OCR:

$$\text{Activate AsyncOCR}\left(I_{\text{crop}}^{(t)}\right) \iff (c_i \in \text{SpeedSign}) \land (w_i > 30 \land h_i > 30) \land \left(\text{Key}(\mathbf{b}_i) \notin \mathcal{C}_{\text{ocr}}\right)$$

2. Điều kiện chọn đèn giao thông chính:

$$\mathbf{b}^* = \arg\max_{\mathbf{b}_i \in \text{TrafficLight}} (w_i \times h_i)$$

3. Logic cập nhật trạng thái đèn theo thời gian ($S_{\text{light}}^{(t)}$):

$$S_{\text{light}}^{(t)} = \begin{cases} 
P^{(t)} & \text{nếu } C_{\text{pending}}^{(t)} \ge 4 \\ 
S_{\text{light}}^{(t-1)} & \text{nếu } 0 < C_{\text{missing}}^{(t)} < 10 \\ 
\emptyset & \text{nếu } C_{\text{missing}}^{(t)} \ge 10 
\end{cases}$$

\- Tăng đếm khớp mẫu: $C_{\text{pending}}^{(t)} = C_{\text{pending}}^{(t-1)} + 1 \quad \text{nếu } \text{TrafficLightColor} = P^{(t-1)}$

\- Tăng đếm mất dấu: $C_{\text{missing}}^{(t)} = C_{\text{missing}}^{(t-1)} + 1 \quad \text{nếu } \text{TrafficLightColor} = \emptyset$

### I. Mô hình ước tính khoảng cách bằng camera đơn

$$d = \frac{H \times \mathrm{ImageHeight}}{2 \times h \times \tan\left(\frac{FOV_v}{2}\right)}$$

\- $d$: Khoảng cách từ camera đến vật thể.

\- $H$: Chiều cao thực tế của vật ngoài đời (mét).

\- $h$: Chiều cao của vật thu được trên ảnh (pixel).

\- $\mathrm{ImageHeight}$: Độ phân giải chiều cao của bức ảnh (pixel) — *ví dụ: 1080p thì $\mathrm{ImageHeight} = 1080$*.

\- $FOV_v$: Góc nhìn theo chiều dọc của camera (Vertical Field of View, tính bằng độ - $^\circ$).

### II. Quy trình nhận diện OCR và phân tích giá trị biển báo giao thông

$$\text{Output}(I_{\mathrm{crop}}) = \begin{cases} 
V + \text{" t"} & \text{nếu } (V \in S_t) \land ('t' \in L \lor V \in S_t) \\ 
V + \text{" m"} & \text{nếu } (V \in S_m) \land ('m' \in L \lor V \in S_m) \\ 
V + \text{" km/h"} & \text{nếu } V \in S_k \\ 
\emptyset & \text{ngược lại} 
\end{cases}$$

Trong đó, giá trị $V$ và $L$ được tính qua chuỗi hàm:

$$V = \text{Correction}\left(\text{Digits}\left(\text{OCR}\left(\mathcal{T}_{120}\left(\mathcal{M}_{\mathrm{Red}}\left(\text{Resize}(I_{\mathrm{crop}}, S)\right)\right)\right)\right)\right)$$

$$L = \text{Letters}\left(\text{OCR}\left(\mathcal{T}_{120}\left(\mathcal{M}_{\mathrm{Red}}\left(\text{Resize}(I_{\mathrm{crop}}, S)\right)\right)\right)\right)$$

\- Hàm phóng tỷ lệ $S$: $S = \max\left(2.0, \frac{80}{\min(h,w)}\right)$

\- Hàm lọc viền đỏ $\mathcal{M}_{\mathrm{Red}}$: Lọc các pixel có màu đỏ trong không gian HSV và đổi sang trắng $(255, 255, 255)$.

\- Hàm phân ngưỡng $\mathcal{T}_{120}$: Binary threshold tại ngưỡng $120$ để biến ảnh xám thành đen/trắng nét hơn.

\- Hàm OCR: Trích xuất các ký tự thuộc tập $A = \{0..9, ., t, m, T, M\}$.

\- Hàm sửa lỗi $\text{Correction}$:

$$V \in \{15, 1.5, 154, 151, 1.54\} \text{ và } ('t' \in L \text{ hoặc xuất hiện '1', '4'}) \implies \begin{cases} V \leftarrow 1.5 \\ L \leftarrow 't' \end{cases}$$

\- Các tập hợp kiểm tra: $S_t$ (Tấn), $S_m$ (Mét), $S_k$ (Tốc độ).

### III. Bộ phân loại trạng thái và màu đèn giao thông dựa trên hệ quy tắc

$$\text{TrafficLightColor}(I, \mathbf{b}) = \underset{c \in \{\text{Red}, \text{Yellow}, \text{Green}\}}{\operatorname{argmax}} \left( \sum_{(x,y) \in I_{\mathrm{crop}}} \mathbf{1}_{\text{Color}_c}(x,y) \right)$$

Với điều kiện ngưỡng kích hoạt điểm tối thiểu:

$$\max_c \left( \sum_{(x,y) \in I_{\mathrm{crop}}} \mathbf{1}_{\text{Color}_c}(x,y) \right) \ge 0.008 \times (H_{\mathrm{crop}} \times W_{\mathrm{crop}})$$

*(Nếu không thỏa mãn ngưỡng trên, trả về None)*

#### Mở rộng biểu thức logic toán học cho từng không gian màu

Cho ảnh cắt $I_{\mathrm{crop}} = I[y_1 : y_2, x_1 : x_2]$ có kích thước $H_{\mathrm{crop}} \times W_{\mathrm{crop}}$:

1. Chuẩn hóa kênh màu (Normalized RGB):

$$I(x,y) = I_{\text{BGR}}(x,y) + 1$$

$$S_{\text{RGB}}(x,y) = R(x,y) + G(x,y) + B(x,y)$$

$$r_n = \frac{R}{S_{\text{RGB}}}, \quad g_n = \frac{G}{S_{\text{RGB}}}, \quad b_n = \frac{B}{S_{\text{RGB}}}$$

2. Điều kiện pixel sáng (Brightness Condition):

$$V(x,y) = \max(R, G, B) > 120$$

3. Hàm chỉ thị vùng màu $\mathbf{1}_{\text{Color}_c}(x,y)$:

$$\mathbf{1}_{\text{Red}}(x,y) = V(x,y) \land \Big( (r_n > 0.42 \land r_n > 1.2g_n) \lor (R > 1.3G \land R > 1.2B) \Big)$$

$$\mathbf{1}_{\text{Yellow}}(x,y) = V(x,y) \land \Big( r_n > 0.35 \land g_n > 0.35 \land |r_n - g_n| < 0.12 \land r_n > 1.3b_n \Big)$$

$$\mathbf{1}_{\text{Green}}(x,y) = V(x,y) \land \Big( (g_n > 0.38 \land g_n > 1.05r_n) \lor (G > 1.1R \land G > 0.9B) \Big)$$




