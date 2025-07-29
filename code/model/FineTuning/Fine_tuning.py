from datasets import Dataset
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer
import numpy as np
import torch
import evaluate # 평가 지표 로드를 위해 추가
from sklearn.metrics import f1_score, roc_auc_score # F1-score, ROC AUC 계산을 위해 추가
import random

# 데이터 로드
original = pd.read_excel('/home/mssggg/CPU/code/api/data/TUNED/train_data(Diffusion model).xlsx')
data = original['특허명칭']
labels_raw = original[['근본', '생성형 AI']].values.tolist()

# 다중 레이블을 위한 float 변환 (np.float32로 변경 권장)
labels = np.array(labels_raw).astype(np.float32).tolist() # Trainer는 torch.FloatTensor를 기대하므로 float32가 더 좋습니다.

# 데이터셋 생성
dataset = Dataset.from_dict({'text': data.tolist(), 'labels': labels}) # Pandas Series를 list로 변환

# 토크나이저와 모델 로드
model_name = "anferico/bert-for-patents"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2, problem_type="multi_label_classification")

# 토큰화 함수
def preprocess_function(examples):
    return tokenizer(examples['text'], truncation=True, padding='max_length', max_length=512)

tokenized_datasets = dataset.map(preprocess_function, batched=True)

# trainer를 위한 데이터셋 포맷 변경
train_test_split_dataset = tokenized_datasets.train_test_split(test_size=0.3, seed=42) # 재현성을 위해 seed 추가

# train, test 데이터셋 각각에 대해 포맷 설정
train_dataset = train_test_split_dataset['train']
eval_dataset = train_test_split_dataset['test']

train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
eval_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

# 시드 고정
seed = 42 # 원하는 아무 정수 값
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed) # GPU를 사용하는 경우 (필수)
torch.backends.cudnn.deterministic = True # CUDA 연산 결정론적 설정
torch.backends.cudnn.benchmark = False # 성능 저하 있을 수 있으나 재현성 확보에 도움



# =========================================================================
# 새로 추가된 부분: 평가 지표 계산 함수
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    # 로짓(logits)에 시그모이드를 적용하여 확률로 변환하고, 0.5를 기준으로 이진 예측
    predictions = (torch.sigmoid(torch.from_numpy(logits)) > 0.5).int().numpy()

    # F1-score (멀티 라벨에서는 micro 또는 macro 평균이 유용)
    f1_micro = f1_score(labels, predictions, average='micro')
    f1_macro = f1_score(labels, predictions, average='macro')

    # ROC AUC (각 라벨별로 계산 후 평균, 모든 라벨이 동일 값일 경우 오류 방지)
    roc_auc = "N/A"
    try:
        roc_auc = roc_auc_score(labels, logits, average='micro') # 로짓 사용
    except ValueError:
        pass

    return {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "roc_auc_micro": roc_auc,
    }
# =========================================================================


# 트레이닝 인자
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch", # eval_strategy로 변경되었는지 확인
    learning_rate=2e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10,
    save_strategy="epoch",           # 매 에포크마다 모델 저장
    load_best_model_at_end=True,     # 학습 종료 시 최고 성능 모델 로드
    metric_for_best_model="f1_macro", # 최고 모델을 결정할 지표 (compute_metrics의 키와 일치)
    greater_is_better=True,          # 해당 지표는 높을수록 좋음
    report_to="none"                 # wandb 등 로깅 시스템 사용하지 않을 경우
)

# Trainer 정의
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset, # 분할된 데이터셋 사용
    eval_dataset=eval_dataset,   # 분할된 데이터셋 사용
    tokenizer=tokenizer,         # 토크나이저 전달 (저장/재로드 시 유용)
    compute_metrics=compute_metrics, # <--- 추가된 부분: 평가 지표 함수 전달
)

# 파인튜닝 시작
print("파인튜닝 시작...")
trainer.train()
print("파인튜닝 완료!")

# 최종 평가 (선택 사항)
final_eval_results = trainer.evaluate()
print(f"최종 평가 결과: {final_eval_results}")