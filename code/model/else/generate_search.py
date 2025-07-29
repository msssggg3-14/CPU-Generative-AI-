import torch
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# ================================================================
# 설정 (Configuration)
# ================================================================
TOP_N = 10                   # 각 개념별로 최대 몇 개의 키워드를 선택할지
SIMILARITY_THRESHOLD = 0.4   # ✅ [진화 1] 키워드를 포함시킬 최소 유사도 점수
OPERATOR = ' NEAR15 '          # ✅ [진화 2] 그룹을 묶는 연산자. ' AND ' 보다 정교한 근접 연산자 사용 (15단어 이내)


# ================================================================
# 모델 및 함수 (기존과 동일)
# ================================================================
model_name = "KIPI-ai/KorPatElectra"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)
model.eval()

@torch.no_grad()
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :]


# ================================================================
# 실행 로직
# ================================================================
main_topic = "Multimodal"
sub_concepts = ["Contrastive Pre-training (CLIP)", "Shared Embedding Space", "Cross-Attention Mechanism", "Multi-Modal Encoders", "Fine-grained Control (e.g., ControlNet)"]

keyword_candidates = list(set([
"Multimodal", "Model", "Modality", "specific", "Encoder", "shared", "Latent", "space", "Corss", "modal",
"Embedding", "Fusion", "attention", "Mechanism", "contrastive", "learning", "CCA", "Canonical",
"Correlation", "Analysis", "Language", "vision",
"다중", "모달리티", "모델", "인코더", "공유", "잠재", "공간", "교차", "모달", "임베딩",
"멀티모달", "융합", "어텐션", "메커니즘", "대조", "학습", "정준", "상관", "분석", "비전", "언어"
]))



# ✅ [진화 3] 키워드 후보군의 임베딩을 '미리 한 번만' 계산하여 효율성 극대화
print("키워드 후보군 임베딩 중... (최초 1회 실행)")
keyword_embeddings = {kw: get_embedding(kw) for kw in keyword_candidates}
print("임베딩 완료.")

print(keyword_embeddings["Multimodal"].shape)



# 개념별 키워드 그룹핑
concept_groups = []
for concept in sub_concepts:
    concept_vec = get_embedding(concept)
    
    ranked_keywords = []
    # 미리 계산된 임베딩을 사용하여 불필요한 반복 계산 방지
    for kw, kw_vec in keyword_embeddings.items():
        score = cosine_similarity(concept_vec, kw_vec).item()
        ranked_keywords.append((kw, score))
    
    ranked_keywords.sort(key=lambda x: x[1], reverse=True)
    
    # [진화 1] 유사도 임계값을 넘는 상위 N개의 키워드만 선택
    top_keywords_for_concept = [
        kw for kw, score in ranked_keywords[:TOP_N] 
        if score >= SIMILARITY_THRESHOLD
    ]
    
    # 유효한 키워드가 있을 경우에만 그룹 생성
    if top_keywords_for_concept:
        concept_group_expr = ' OR '.join(f'"{kw}"' for kw in top_keywords_for_concept)
        concept_groups.append(f"({concept_group_expr})")

# [진화 2] 설정된 연산자로 최종 검색식 결합
final_search_expr = OPERATOR.join(concept_groups)

# 결과 출력
print("\n📌 [검색 주제]")
print(f"{main_topic}")

print("\n📊 [개념별 핵심 키워드 그룹]")
for i, concept in enumerate(sub_concepts):
    # 생성된 그룹이 있는 경우에만 출력
    if i < len(concept_groups):
        print(f"- {concept}: {concept_groups[i]}")

print("\n🔍 [진화된 Boolean 검색식]")
print(final_search_expr)