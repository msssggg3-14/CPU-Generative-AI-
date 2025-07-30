import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
# --- 설정값 ---
fine_tuned_model_path = '/home/mssggg/CPU/results/checkpoint-45'
all_possible_labels = ['근본', '생성형 AI']
prediction_threshold = 0.5
inference_batch_size = 32  # 배치 크기 설정 (메모리 상황에 따라 조정 가능)
# ------------------

tokenizer = AutoTokenizer.from_pretrained(fine_tuned_model_path)
model = AutoModelForSequenceClassification.from_pretrained(fine_tuned_model_path)
model.eval()

# GPU 사용 가능 시 모델을 GPU로 이동
if torch.cuda.is_available():
    model.to("cuda")
    print("모델이 GPU로 로드되었습니다.")
else:
    print("GPU를 사용할 수 없어 모델이 CPU로 로드되었습니다.")


new_data_df = pd.read_excel('code/api/data/TUNED/foreign_patents_unique(VAE).xlsx')
print(new_data_df.head())
new_data_df.dropna(subset=['특허명칭'], inplace=True)
print(new_data_df.head())
print(f"\n새로운 데이터 {len(new_data_df)}개를 불러왔습니다.")

def preprocess_for_inference(texts):
    return tokenizer(texts, truncation=True, padding='max_length', max_length=256, return_tensors="pt")

tokenized_inputs = preprocess_for_inference(new_data_df['특허명칭'].tolist())

# GPU 사용 시 입력 텐서도 GPU로 이동
if torch.cuda.is_available():
    tokenized_inputs = {k: v.to("cuda") for k, v in tokenized_inputs.items()}

# 예측 수행
print("모델이 예측을 수행 중입니다...")
with torch.no_grad(): # 예측 시에는 그래디언트 계산을 비활성화 (메모리, 속도 최적화)
    outputs = model(**tokenized_inputs)
    logits = outputs.logits # 모델의 예측 로짓(raw scores)을 가져옵니다.

# 예측값 후처리
probabilities = torch.sigmoid(logits).cpu().numpy() # GPU -> CPU -> NumPy 변환

# 예측된 라벨 
predicted_labels_binary = (probabilities >= prediction_threshold).astype(int)


# 사람이 읽을 수 있는 라벨 이름으로 변환
predicted_labels_named = []
for sample_labels in predicted_labels_binary:
    current_sample_labels = []
    for i, label_value in enumerate(sample_labels):
        if label_value == 1:
            current_sample_labels.append(all_possible_labels[i])
    # 예측된 라벨이 하나도 없을 경우 '(라벨 없음)'으로 표시
    predicted_labels_named.append(current_sample_labels if current_sample_labels else ["(라벨 없음)"])

# 결과를 원본 DataFrame에 추가
new_data_df['predicted_probabilities'] = probabilities.tolist()
new_data_df['predicted_labels_binary'] = predicted_labels_binary.tolist()
new_data_df['predicted_labels_named'] = predicted_labels_named

# 각 라벨 컬럼을 이진 형식으로 추가 (필요한 경우)
for i, label_name in enumerate(all_possible_labels):
    new_data_df[f'predicted_{label_name}'] = predicted_labels_binary[:, i]


print("\n--- 분류(라벨링) 결과 ---")
print(new_data_df)

# 결과를 새로운 Excel 파일로 저장 (선택 사항)
output_excel_path = "./labeled_new_data(Diffusion model).xlsx"
new_data_df.to_excel(output_excel_path, index=False)
print(f"\n분류 결과가 '{output_excel_path}' 파일로 저장되었습니다.")
