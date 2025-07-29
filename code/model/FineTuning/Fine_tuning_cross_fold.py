import pandas as pd
import numpy as np
import torch
import random
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# =========================================================================
# 1. 시드 고정 및 기본 설정
# =========================================================================
def set_seed(seed: int):
    """실행 결과를 재현하기 위해 시드를 고정합니다."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SEED = 42
set_seed(SEED)
MODEL_NAME = "anferico/bert-for-patents"

# =========================================================================
# 2. 데이터 로드 및 분할
# =========================================================================
print("데이터 로드 및 분할 중...")
original_df = pd.read_excel('/home/mssggg/CPU/code/api/data/TUNED/train_data(Diffusion model).xlsx')

# 텍스트와 레이블 준비
data = original_df['특허명칭'].tolist()
labels = np.array(original_df[['근본', '생성형 AI']].values).astype(np.float32)

# 데이터를 훈련+검증 세트와 테스트 세트로 먼저 분리 (예: 80% / 20%)
train_val_texts, test_texts, train_val_labels, test_labels = train_test_split(
    data, labels, test_size=0.2, random_state=SEED, stratify=labels[:,0] # 레이블 분포 고려
)

# 훈련+검증 세트를 다시 훈련 세트와 검증 세트로 분리 (예: 원래 데이터의 60% / 20%)
train_texts, val_texts, train_labels, val_labels = train_test_split(
    train_val_texts, train_val_labels, test_size=0.25, random_state=SEED, stratify=train_val_labels[:,0] # 0.25 * 0.8 = 0.2
)

# Hugging Face Dataset 형식으로 변환
train_dataset = Dataset.from_dict({'text': train_texts, 'labels': train_labels.tolist()})
val_dataset = Dataset.from_dict({'text': val_texts, 'labels': val_labels.tolist()})
test_dataset = Dataset.from_dict({'text': test_texts, 'labels': test_labels.tolist()})

print(f"훈련 데이터: {len(train_dataset)}개")
print(f"검증 데이터: {len(val_dataset)}개")
print(f"테스트 데이터: {len(test_dataset)}개")


# =========================================================================
# 3. 토크나이저 및 모델 준비
# =========================================================================
print("토크나이저 및 모델 로드 중...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    problem_type="multi_label_classification"
)

# 토큰화 함수
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

# 데이터셋 토큰화
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)
tokenized_test = test_dataset.map(preprocess_function, batched=True) # 테스트셋도 미리 토큰화


# =========================================================================
# 4. 훈련 설정 및 평가 함수
# =========================================================================
# 훈련 중 검증에 사용할 평가 함수 (임계값 0.5 고정)
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 시그모이드 및 0.5 임계값으로 예측
    predictions = (torch.sigmoid(torch.from_numpy(logits)) > 0.5).int().numpy()

    f1_macro = f1_score(labels, predictions, average='macro', zero_division=0)
    roc_auc = roc_auc_score(labels, logits, average='macro')

    return {
        "f1_macro": f1_macro,
        "roc_auc_macro": roc_auc,
    }

# Trainer 인자 설정
training_args = TrainingArguments(
    output_dir="./results_final",
    # --- 평가 및 저장 전략 ---
    eval_strategy="epoch",
    save_strategy="epoch",
    # --- 최적 모델 선택 ---
    load_best_model_at_end=True,
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    # --- 학습 파라미터 ---
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    # --- 로깅 ---
    logging_dir='./logs_final',
    logging_steps=10,
    report_to="none"
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,  # **검증 세트**를 사용하여 최적 모델 선택
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)


# =========================================================================
# 5. 모델 훈련
# =========================================================================
print("\n모델 훈련 시작...")
trainer.train()
print("모델 훈련 완료!")


# =========================================================================
# 6. 최적 임계값 탐색 (검증 세트 사용)
# =========================================================================
print("\n최적 임계값 탐색 시작 (검증 데이터 사용)...")
# 검증 세트에 대한 예측(logits) 얻기
val_predictions = trainer.predict(tokenized_val)
val_logits = val_predictions.predictions

best_f1 = 0
best_threshold = 0.5
sigmoid = torch.nn.Sigmoid()

# 0.1부터 0.9까지 다양한 임계값을 시도하여 최적의 값을 찾음
for threshold in np.arange(0.1, 0.9, 0.01):
    preds = (sigmoid(torch.from_numpy(val_logits)) > threshold).int().numpy()
    f1 = f1_score(val_labels, preds, average='macro', zero_division=0)
    if f1 > best_f1:
        best_f1 = f1
        best_threshold = threshold

print(f"최적 임계값 발견: {best_threshold:.4f}")
print(f"검증 세트에서의 최고 F1-macro 점수: {best_f1:.4f}")


# =========================================================================
# 7. 최종 성능 평가 (테스트 세트 사용)
# =========================================================================
print("\n최종 모델 성능 평가 시작 (테스트 데이터 사용)...")
# 테스트 세트에 대한 예측(logits) 얻기
test_predictions_output = trainer.predict(tokenized_test)
test_logits = test_predictions_output.predictions

# **최적 임계값**을 사용하여 최종 예측 수행
final_predictions = (sigmoid(torch.from_numpy(test_logits)) > best_threshold).int().numpy()

# 최종 점수 계산
final_f1_micro = f1_score(test_labels, final_predictions, average='micro', zero_division=0)
final_f1_macro = f1_score(test_labels, final_predictions, average='macro', zero_division=0)
final_roc_auc_micro = roc_auc_score(test_labels, test_logits, average='micro')
final_roc_auc_macro = roc_auc_score(test_labels, test_logits, average='macro')

print("\n--- 최종 성능 평가 결과 (Test Set) ---")
print(f"사용한 임계값: {best_threshold:.4f}")
print(f"F1-Score (Micro): {final_f1_micro:.4f}")
print(f"F1-Score (Macro): {final_f1_macro:.4f}")
print(f"ROC-AUC (Micro): {final_roc_auc_micro:.4f}")
print(f"ROC-AUC (Macro): {final_roc_auc_macro:.4f}")
print("-----------------------------------------")