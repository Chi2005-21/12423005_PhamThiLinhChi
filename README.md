# 💸💸DỰ ĐOÁN RỦI RO TÍN DỤNG
## Credit Risk Prediction using Machine Learning
### 1️. Giới thiệu đề tài
#### 1.1 Bài toán

Trong lĩnh vực tài chính – ngân hàng, rủi ro tín dụng là khả năng khách hàng không thể hoặc không sẵn sàng hoàn trả khoản vay đúng hạn. Việc đánh giá sai rủi ro có thể gây ra tổn thất lớn cho tổ chức cho vay.

Đề tài này tập trung xây dựng hệ thống Machine Learning nhằm:

Phân tích dữ liệu khách hàng vay vốn

Dự đoán khả năng vỡ nợ (default) của khách hàng

Hỗ trợ ra quyết định trong xét duyệt tín dụng

#### 1.2 Mục tiêu đề tài

Hiểu và phân tích bộ dữ liệu rủi ro tín dụng

Thực hiện tiền xử lý dữ liệu một cách có hệ thống

Xây dựng và so sánh nhiều mô hình Machine Learning

Đánh giá mô hình bằng các chỉ số phù hợp

Triển khai pipeline huấn luyện và dự đoán (inference)
### 2️. Giới thiệu bộ dữ liệu (Credit Risk Dataset)
#### 2.1 Nguồn dữ liệu

Bộ dữ liệu được lấy từ Kaggle:

🔗 Credit Risk Dataset
https://www.kaggle.com/datasets/laotse/credit-risk-dataset

⚠️ Do dung lượng và điều khoản sử dụng của Kaggle, dữ liệu không được đưa lên GitHub.
Hướng dẫn tải và sử dụng dữ liệu được trình bày trong file:
data/README.md
#### 2.2 Mô tả các thuộc tính dữ liệu
Bộ dữ liệu gồm các thông tin liên quan đến đặc điểm cá nhân, lịch sử tín dụng và khoản vay của khách hàng.

+ person_age: Tuổi của khách hàng
+ person_income: Thu nhập hàng năm (USD)
+ person_home_ownership: Hình thức sở hữu nhà (RENT/OWN/MORTGAGE...)
+ person_emp_length: Số năm làm việc
+ loan_intent: Mục đích vay (PERSONAL, MEDICAL, EDUCATION...)
+ loan_grade: Xếp hạng tín dụng của khoản vay (A–G)
+ loan_amnt: Số tiền vay
+ loan_int_rate: Lãi suất vay (%)
+ loan_status: 1 = rủi ro (default), 0 = tốt
+ loan_percent_income: Tỷ lệ tiền vay / thu nhập
+ cb_person_default_on_file: Từng vỡ nợ (Y/N)
+ cb_person_cred_hist_length: Thời gian lịch sử tín dụng (năm)
##### Nhận xét tổng quan về cấu trúc dữ liệu
Bộ dữ liệu có 32.581 dòng và 12 cột, bao gồm cả đặc trưng dạng số (numerical) và đặc trưng dạng phân loại (categorical).

+ Một số cột chứa giá trị bị thiếu, đặc biệt là person_emp_length và loan_int_rate, cần được xử lý trước khi đưa vào mô hình dự báo rủi ro.
+ Categorical Features: person_home_ownership, loan_intent, loan_grade, cb_person_default_on_file → Các cột này thể hiện loại nhà ở, mục đích vay, xếp hạng khoản vay và trạng thái mặc định trước đó.
+ Binary Numerical Features: loan_status (target) → Biến mục tiêu nhị phân (0 = không vỡ nợ, 1 = vỡ nợ).
+ Continuous Numerical Features: person_age, person_income, person_emp_length, loan_amnt, loan_int_rate, loan_percent_income, cb_person_cred_hist_length → Đây là các biến số liên tục phản ánh đặc điểm tài chính, hành vi tín dụng và mức rủi ro kinh tế của người vay.
**Biến số liên tục có ngoại lai mạnh**
+ person_income: max lên đến 6.000.000 trong khi median chỉ ~55.000 → xuất hiện giá trị ngoại lai lớn, cần kiểm tra và xử lý (log-transform hoặc capping).
+ person_emp_length: max 123 tháng (≈10 năm) nhưng tuổi trung bình chỉ 27 → khả năng có giá trị bất thường.
+ person_age: max 144 là bất hợp lý → rõ ràng outlier, cần làm sạch.

**Biến lãi suất vay (loan_int_rate)**
Trung bình ~11%

Dao động từ 5.42 đến 23.22 → khoảng hợp lý → Tuy nhiên có ~3.000 giá trị bị thiếu, cần xử lý (mean/median hoặc mô hình dự đoán).

**Biến target (loan_status)**
Mean = 0.218 → tỷ lệ "rủi ro" ≈ 21.8% → Dữ liệu mất cân bằng lớp, cần dùng kỹ thuật xử lý như class_weight hoặc SMOTE.

**Biến loan_percent_income**
Trung bình 0.17, max 0.83 → một số người vay gần 80% thu nhập → tiềm ẩn rủi ro cao.

**cb_person_cred_hist_length**
Trung bình 5.8, max 30 → phân phối khá rộng, biểu hiện lịch sử tín dụng không đồng nhất.
### 3️. Tiền xử lý dữ liệu (Data Preprocessing)

Tiền xử lý dữ liệu là bước quan trọng nhằm:

+ Đảm bảo dữ liệu sạch, nhất quán
+ Giúp mô hình học hiệu quả hơn
+ Giảm nhiễu và sai lệch trong huấn luyện
+ Các kỹ thuật tiền xử lý được sử dụng trong đề tài:

#### 3.1 Xử lý giá trị thiếu
Qua phân tích cấu trúc dữ liệu, một số biến số liên tục như person_emp_length và loan_int_rate chứa giá trị bị thiếu. Các mô hình học máy, đặc biệt là Logistic Regression, không thể xử lý trực tiếp các giá trị null. Việc loại bỏ các dòng dữ liệu bị thiếu có thể làm mất thông tin quan trọng do kích thước dữ liệu lớn.

Cách xử lý: Các biến số liên tục được điền bằng giá trị trung vị (median) nhằm giảm ảnh hưởng của ngoại lai và giữ nguyên phân phối dữ liệu.

#### 3.2 Xử lý biến phân loại

Bộ dữ liệu chứa nhiều biến dạng phân loại, trong khi các mô hình học máy chỉ có thể làm việc với dữ liệu số. Ngoài ra, các biến phân loại có bản chất khác nhau: có biến có thứ tự (ordinal), có biến không có thứ tự (nominal), và có biến nhị phân.

Cách xử lý

Binary mapping cho biến nhị phân
Mã hóa biến có thứ tự
One-Hot Encoding cho biến không có thứ tự
#### 3.3 Chuẩn hóa dữ liệu số
**Lý do cần chuẩn hóa**

Các biến số liên tục trong bộ dữ liệu có thang đo rất khác nhau, ví dụ:

+ person_income: hàng chục nghìn USD
+ loan_amnt: vài nghìn đến vài chục nghìn
+ loan_int_rate: đơn vị phần trăm
+ cb_person_cred_hist_length: đơn vị năm
Qua bước trực quan hóa, có thể thấy phần lớn các biến này phân phối lệch phải và chênh lệch về độ lớn. Điều này có thể ảnh hưởng tiêu cực đến các mô hình nhạy cảm với thang đo, đặc biệt là Logistic Regression, khiến mô hình hội tụ chậm hoặc học lệch về các biến có giá trị lớn.

Do đó, cần chuẩn hóa dữ liệu để các biến số liên tục có cùng thang đo, giúp mô hình học hiệu quả và ổn định hơn.

**Cách xử lý**

+ Áp dụng Standard Scaling cho các biến số liên tục:
+ Đưa dữ liệu về phân phối có mean = 0 và std = 1
+ Thực hiện chuẩn hóa thông qua ColumnTransformer: Đảm bảo quy trình tiền xử lý nhất quán giữa tập huấn luyện và tập kiểm tra
#### 3.4  Xử lý mất cân bằng lớp (Imbalanced Data)
**Lý do cần xử lý**

Biến mục tiêu loan_status có sự mất cân bằng rõ rệt, trong đó nhóm khách hàng rủi ro (default = 1) chiếm tỷ lệ thấp hơn đáng kể. Nếu không xử lý, mô hình sẽ có xu hướng ưu tiên dự đoán lớp an toàn (0), dẫn đến bỏ sót nhiều trường hợp rủi ro (False Negative), điều này không phù hợp trong bài toán đánh giá tín dụng.

**Phương pháp xử lý**

Trong bước tiền xử lý, vấn đề mất cân bằng lớp được xử lý bằng gán trọng số cho lớp thiểu số, nhằm tăng mức độ phạt khi mô hình dự đoán sai khách hàng rủi ro.
+ Với các mô hình truyền thống (Logistic Regression, Decision Tree, Random Forest), sử dụng class_weight = "balanced" để tự động điều chỉnh trọng số theo tỷ lệ hai lớp.
+ Với LightGBM, sử dụng scale_pos_weight, được tính dựa trên tỷ lệ giữa số mẫu không rủi ro và rủi ro, giúp tăng ảnh hưởng của lớp thiểu số trong quá trình tối ưu hàm mất mát.
+ Việc điều chỉnh ngưỡng dự đoán (decision threshold) sẽ được thực hiện ở bước huấn luyện và đánh giá mô hình, không áp dụng trực tiếp trong giai đoạn tiền xử lý.
### 4️. Pipeline huấn luyện & dự đoán

Toàn bộ quy trình được xây dựng theo pipeline thống nhất:

Dữ liệu gốc -> Tiền xử lý dữ liệu -> Chia train / test -> Huấn luyện mô hình -> Đánh giá mô hình -> Lưu mô hình tốt nhất -> Inference (dự đoán dữ liệu mới)

**Pipeline giúp:**

+ Tái sử dụng dễ dàng
+ Tránh data leakage
+ Thuận tiện cho triển khai thực tế

Pipeline được xây dựng bằng scikit-learn Pipeline kết hợp với ColumnTransformer nhằm gom toàn bộ các bước tiền xử lý và huấn luyện mô hình vào một quy trình thống nhất. Nhờ đó, các bước như chuẩn hóa dữ liệu, mã hóa biến phân loại và huấn luyện mô hình chỉ được học từ tập huấn luyện, sau đó áp dụng lại cho tập validation và test theo cùng một cách.

Cách tiếp cận này giúp tránh hiện tượng rò rỉ dữ liệu (data leakage), đồng thời đảm bảo tính nhất quán giữa các tập dữ liệu. Ngoài ra, pipeline còn giúp việc lưu mô hình và tái sử dụng cho dự đoán dữ liệu mới (inference) trở nên đơn giản và thuận tiện hơn.

### 5️. Mô hình sử dụng

Các mô hình Machine Learning được thử nghiệm:

🔹 **Logistic Regression **
+ Mô hình tuyến tính, đơn giản
+ Dễ diễn giải kết quả
+ Phù hợp với bài toán phân loại nhị phân

🔹 **Decision Tree**
+ Mô hình phi tuyến
+ Dễ hiểu, trực quan
+ Tuy nhiên dễ overfitting

🔹 **Random Forest**

+ Tập hợp nhiều Decision Tree
+ Giảm overfitting
+ Hoạt động tốt với dữ liệu tabular

🔹 **LightGBM**

+ Mô hình Gradient Boosting hiện đại
+ Huấn luyện nhanh
+ Hiệu quả cao với dữ liệu lớn
+ Thường cho kết quả tốt nhất trong bài toán tín dụng

👉 Mô hình có hiệu quả tốt nhất sẽ được lựa chọn và lưu để sử dụng cho inference.
### 6️. Đánh giá mô hình (Evaluation Metrics)

Các chỉ số đánh giá được sử dụng:

+ Accuracy: Tỷ lệ dự đoán đúng tổng thể
+ Precision: Mức độ chính xác khi dự đoán khách hàng rủi ro
+ Recall: Khả năng phát hiện đúng khách hàng rủi ro
+ F1-score: Trung bình điều hòa giữa Precision và Recall
+ Confusion Matrix: Phân tích chi tiết đúng/sai
+ ROC-AUC: Đánh giá khả năng phân biệt hai lớp

📌 Trong bài toán tín dụng, Recall và AUC đặc biệt quan trọng để hạn chế bỏ sót khách hàng rủi ro.
### 7. Hướng dẫn cài đặt & chạy dự án
#### 7.1 Cài đặt môi trường
**pip install -r requirements.txt**

#### 7.2 Huấn luyện mô hình
**python app/train.py**

Mô hình sau khi huấn luyện sẽ được lưu tại:
**models/lgbm_pipeline.pkl.gz**

#### 7.3 Chạy demo 
Demo trong:
**python demo/app.py**
### 8️. Cấu trúc thư mục dự án
```
myproject/
├── app/
│ ├── _init_.py
│ ├── data_analysis.py
│ ├── preprocess.py
│ ├── models.py
│ ├── train.py
│ └── evaluate.py
├── models/
│ └── lgbm_pipeline.pkl.gz
├── data/
│ └── README.md
| └── sample_credit_risk.csv
├── demo/
├── reports/ BaoCao_ML LinhChi.pdf
├── slides/BaoCao_ML_Chi.pdf
├── README.md
├── requirements.txt
└── .gitignore
```

### 9️. Tác giả
Họ tên: Phạm Thị Linh Chi

Mã sinh viên: 12423005

Lớp: 124231
