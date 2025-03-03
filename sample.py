import streamlit as st
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import BingSearchAPIWrapper
from langgraph.graph import StateGraph
from typing import List, Dict, Any, Optional
from pydantic import BaseModel
from dotenv import load_dotenv
import os

# .envファイルから環境変数を読み込む
load_dotenv()

# 環境変数の取得
AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
BING_SUBSCRIPTION_KEY = os.getenv("BING_SUBSCRIPTION_KEY")
BING_SEARCH_URL = os.getenv("BING_SEARCH_URL")

# モデルと検索APIの初期化
model = AzureChatOpenAI(
    api_key=AZURE_OPENAI_API_KEY,
    endpoint=AZURE_OPENAI_ENDPOINT,
    deployment_name=AZURE_OPENAI_DEPLOYMENT_NAME
)

search = BingSearchAPIWrapper(
    bing_subscription_key=BING_SUBSCRIPTION_KEY,
    bing_search_url=BING_SEARCH_URL
)

# 入力データの型定義
class InputData(BaseModel):
    research_theme: str
    sections: Optional[List[str]] = None
    summaries: Optional[Dict[str, str]] = None
    references: Optional[Dict[str, List[Dict[str, str]]]] = None

# 調査セクションの自動生成
def generate_sections(state: InputData) -> InputData:
    prompt = ChatPromptTemplate.from_template(
        f"""
        調査テーマ: {state.research_theme}
        このテーマについて、適切な観点を考慮し、詳細なセクションを5〜7個提案してください。
        各セクションは、1. 市場動向 のようにリスト形式で出力してください。
        """
    )
    chain = prompt | model | StrOutputParser()
    response = chain.invoke({})

    # セクションリストの整形
    sections = [line.split(". ")[1] for line in response.split("\n") if ". " in line]
    state.sections = sections if sections else ["概要", "詳細分析", "結論"]
    return state

# 検索と要約処理
def process_section(state: InputData) -> InputData:
    summaries = {}
    references = {}

    for section in state.sections:
        st.write(f"Processing section: {section}")

        # 検索キーワード生成
        prompt = ChatPromptTemplate.from_template(
            f"""
            調査テーマ: {state.research_theme}
            セクション: {section}
            上記に基づき、詳細な検索キーワードを3つ生成してください。
            """
        )
        chain = prompt | model | StrOutputParser()
        keywords = chain.invoke({})

        # 情報検索
        search_results = search.results(keywords, num_results=10)
        references[section] = [{"title": res["name"], "link": res["url"]} for res in search_results]

        # 情報統合と要約
        summary_prompt = ChatPromptTemplate.from_template(
            """
            あなたは専門的なリサーチャーです。
            以下の情報をもとに、詳細なビジネスレポートを作成してください。

            **構成**
            1. **序論**: セクションの概要を説明する
            2. **本論**: 検索結果の詳細な分析（市場動向・競合情報・リスクなど）
            3. **結論**: 調査から得られる示唆や推奨事項

            **参考データ**
            ---
            {results}
            ---

            **最終レポート**
            """
        )
        chain = summary_prompt | model.bind(temperature=0.3, max_tokens=10000) | StrOutputParser()
        summaries[section] = chain.invoke({"results": search_results})

    state.summaries = summaries
    state.references = references
    return state

# 最終レポート統合
def compile_report(state: InputData) -> InputData:
    final_report_prompt = ChatPromptTemplate.from_template(
        """
        あなたは専門的なリサーチャーです。
        以下の各セクションの要約をもとに、統合されたビジネスレポートを作成してください。

        **構成**
        1. **序論**: 調査テーマの背景と目的を説明
        2. **本論**: 各セクションの要約を統合し、論理的な流れで分析
        3. **結論**: 全体を通した示唆や推奨事項

        **セクション別要約**
        ---
        {summaries}
        ---

        **最終レポート**
        """
    )
    chain = final_report_prompt | model.bind(temperature=0.3, max_tokens=10000) | StrOutputParser()
    state.summaries["final_report"] = chain.invoke({"summaries": state.summaries})
    return state

# LangGraphワークフロー定義
workflow = StateGraph(InputData)
workflow.add_node("generate_sections", generate_sections)
workflow.add_node("process_section", process_section)
workflow.add_node("compile_report", compile_report)
workflow.add_edge("generate_sections", "process_section")
workflow.add_edge("process_section", "compile_report")
workflow.set_entry_point("generate_sections")

# ワークフローをコンパイル
workflow = workflow.compile()

# Streamlitアプリの構築
st.title("調査レポート自動生成アプリ")
research_theme = st.text_input("調査テーマを入力してください")

if st.button("レポート生成"):
    if research_theme:
        input_data = InputData(research_theme=research_theme)
        output = workflow.invoke(input_data)

        if isinstance(output, dict):  # LangGraph は辞書形式で返す
            st.write("Output data:", output)  # デバッグ用

            # 統合レポートの表示
            final_report = output.get("summaries", {}).get("final_report", "レポート生成に失敗しました。")
            st.header("最終ビジネスレポート")
            st.write(final_report)

            # 参考文献の表示
            references = output.get("references", {})
            st.subheader 