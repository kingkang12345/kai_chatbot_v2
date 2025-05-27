import os
import pandas as pd
from langchain_teddynote.document_loaders import HWPLoader
from langchain_openai import ChatOpenAI
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain.prompts import ChatPromptTemplate

# 1. 요약할 폴더 지정
FOLDER_PATH = "./hwpsummary/data" # 원하는 폴더로 변경

# 2. LLM 준비 (OpenAI API 키 필요)
llm = ChatOpenAI(
    temperature=0,
    openai_api_key="sk-k7ZAoJmlclL75pjwHgEcFw",
    openai_api_base = "https://genai-sharedservice-americas.pwcinternal.com/",
    model_name = "openai.gpt-4.1-2025-04-14"
)

# 프롬프트 불러오기
# map_prompt = hub.pull("teddynote/map-prompt")
custom_map_prompt = ChatPromptTemplate.from_template(
    """
    Your task is to extract the main thesis and important details from the given document. Answer should be in the same language as the given document.

    #Format:
    - thesis 1
    - thesis 2
    - thesis 3
    - ...

    Here is a given document:
    {doc}

    Write as many sentences as needed to cover all important details. Be as detailed as possible.
    #Answer:
    """
)
reduce_prompt = hub.pull("teddynote/reduce-prompt")

# map chain
map_chain = custom_map_prompt | llm | StrOutputParser()

# reduce chain
reduce_chain = reduce_prompt | llm | StrOutputParser()

# 요약 결과를 저장할 리스트
summaries = []

# 3. 폴더 내 모든 .hwp 파일 처리
for filename in os.listdir(FOLDER_PATH):
    if filename.endswith(".hwp"):
        file_path = os.path.join(FOLDER_PATH, filename)
        loader = HWPLoader(file_path)
        docs = loader.load()
        # map 단계: 각 문서 chunk에 대해 요약
        map_inputs = [{"doc": doc.page_content, "language": "Korean"} for doc in docs]
        doc_summaries = map_chain.batch(map_inputs)
        # reduce 단계: 요약들을 하나로 합침
        reduce_input = {
            "doc_summaries": "\n".join(doc_summaries),
            "language": "Korean"
        }
        summary = reduce_chain.invoke(reduce_input)
        # 리스트에 결과 추가
        summaries.append({"파일명": filename, "요약": summary})

# 모든 요약 결과를 하나의 엑셀 파일로 저장
if summaries:
    df = pd.DataFrame(summaries)
    excel_path = os.path.join(FOLDER_PATH, "HWP_Summaries.xlsx")
    df.to_excel(excel_path, index=False)