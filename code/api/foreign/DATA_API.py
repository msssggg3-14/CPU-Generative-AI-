import requests
import xmltodict
import pandas as pd
from pathlib import Path
import time

# =================================================================
# ▼ 1. 검색 조건 및 저장 경로 설정
# =================================================================
SEARCH_KEYWORD = 'VAE AND Generative Ai' # Generative Adversarial Networks 다음에 하기
SEARCH_RANGE_YEARS = "10"
COUNTRIES = "US"
SORT_FIELD = "AD"
SORT_STATE = "desc"
API_KEY = "JgtgArUNZtqsQ5aasC5AMfuKl5rbwa29LRW90wHV2Cs="

# 엑셀 저장 폴더 (원하는 경로로 수정)
SAVE_FOLDER = Path(r"code/api/data")
SAVE_FOLDER.mkdir(parents=True, exist_ok=True)
excel_path = SAVE_FOLDER / "foreign_patents.xlsx"
# ================================================================s=

# 요청 URL
base_url = "http://plus.kipris.or.kr/openapi/rest/ForeignPatentGeneralSearchService/wordSearch"

current_page = 1
results_list = []

while True:
    params = {
        "searchWord": SEARCH_KEYWORD,
        "searchWordRange": SEARCH_RANGE_YEARS,
        "collectionValues": COUNTRIES,
        "sortField": SORT_FIELD,
        "sortState": SORT_STATE,
        "numOfRows": "30",
        "currentPage": str(current_page),
        "accessKey": API_KEY,
    }

    try:
        response = requests.get(base_url, params=params)
        print(f"\n📡 요청 페이지: {current_page}")
        print("요청 URL:", response.url)
        print("응답 상태:", response.status_code)

        data = xmltodict.parse(response.content)

        items = data.get('response', {}).get('body', {}).get('items', {}).get('searchResult', [])
        if not items:
            print("📭 더 이상 결과 없음. 종료합니다.")
            break
        if not isinstance(items, list):
            items = [items]

        print(f"✅ {len(items)}건 수집됨\n")

        for idx, item in enumerate(items, 1):
            global_index = (current_page - 1) * len(items) + idx
            patent_data = {
                "순번": global_index,
                "특허명칭": item.get('inventionName', '제목 없음'),
                "출원번호": item.get('applicationNo', ''),
                "출원인": item.get('applicant', ''),
                "출원일자": item.get('applicationDate', ''),
                "IPC": item.get('ipc', '')
            }
            results_list.append(patent_data)

            # 콘솔 출력
            print(f"▶ [{patent_data['순번']}] {patent_data['특허명칭']}")
            print(f"   출원번호: {patent_data['출원번호']}")
            print(f"   출원인: {patent_data['출원인']}")
            print(f"   출원일자: {patent_data['출원일자']}")
            print(f"   IPC: {patent_data['IPC']}")
            print("-" * 80)

        current_page += 1
        time.sleep(0.1)


    except Exception as e:
        print("❌ 오류 발생:", e)
        break

# 엑셀 저장
if results_list:
    df = pd.DataFrame(results_list)
    df.to_excel(excel_path, index=False)
    print(f"\n✅ 엑셀 파일 저장 완료: '{excel_path}'")
else:
    print("\n❌ 저장할 특허 데이터가 없습니다.")
