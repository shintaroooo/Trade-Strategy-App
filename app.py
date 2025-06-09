import streamlit as st
st.set_page_config(page_title="AI Entry Lab", layout="centered")

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import HumanMessage
import base64
import os
from dotenv import load_dotenv
from datetime import datetime

# --- セッション状態の初期化（最初にまとめて） ---
if "strategy_history" not in st.session_state:
    st.session_state.strategy_history = {}
if "calendar_notes" not in st.session_state:
    st.session_state.calendar_notes = {}

# --- 環境変数の読み込み ---
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# --- OpenAI モデルの準備 ---
llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0.2, openai_api_key=openai_api_key)

# --- プロンプトテンプレート ---
prompt_template = ChatPromptTemplate.from_template("""
あなたは視覚認識と戦略構築に優れたプロのトレードアナリストです。
以下の画像と情報をもとに、優位性のあるトレード戦略を最大3つまで提案せよ。

【目的】
初心者でも再現できる形で、勝率とリスクリワードが優れたエントリー戦略を提示する。

【与えられる情報】
- チャート画像（最大3枚）
- 通貨ペアまたは銘柄名：{ticker}
- ユーザーのトレードスタイル：{style}

【出力形式】
1. 現在の相場環境：
   - トレンド or レンジ（どちらの局面か）
   - MACD、RSI、ストキャス、BB、ローソク足の構造
   - 上位時間足（H1・H4）の影響が強いか、短期足（M5）が主導か

2. 優位性ある戦略の比較表：

| 戦略名 | 概要 | 勝率予想 | RR比 | エントリー | 損切り | 利確 | 時間足 |
|--------|------|----------|------|------------|--------|--------|--------|
| 戦略A  | ...  | ...      | ...  | ...        | ...    | ...    | ...    |
| 戦略B  | ...  | ...      | ...  | ...        | ...    | ...    | ...    |

3. 各戦略の補足解説：
- なぜそのタイミングが有効なのか
- チャート画像上の根拠（例：5分足の3本前陽線の高値）
- 使用インジケーターの簡単な説明（例：MACDとは何か）

【大原則】
 1. 上位時間足（H1・H4）のトレンドを優先的に分析する。
2. M5での逆張り戦略は、以下の条件をすべて満たす場合に限り提案可：
   - MACDヒストグラムの明確な減速
   - 陰線包み・上ヒゲ・ダブルトップなどの反転足
   - ボリンジャーバンドが拡張から収束傾向にある
3. BBタッチやストキャス80超えのみを根拠とした逆張りは禁止。
4. 各戦略には以下を含める：エントリー条件、損切り、利確、使用時間足、時間足間の整合性評価。
5. 初心者向けにインジケーターの解説と、損切りが守れない心理的対策も補足する。
【ルール】                                                                                          
- 画像は必ず3枚まで使用し、各戦略に関連するものを選定せよ。
- 相場環境（トレンド／レンジ）を明確に分類せよ。
- 戦略の根拠となる時間足を明記せよ（例：M5/H1/H4など）。
- 下位足（例：M5）の戦略を提案する際は、上位足（例：H1・H4）のトレンドと整合していることを確認し、その根拠を明記せよ。
- 上位足が上昇トレンドにある場合（MACDがプラス圏、ローソクが高値更新など）、M5での逆張り（売り）は原則禁止とする。
- 例外的に逆張りを提案する場合は、次のすべての条件を満たすこと：
  - MACDヒストグラムの明確な減速
  - 陰線包み・上髭・ダブルトップなど明確な反転足パターン
  - ボリンジャーバンドが拡張から収束へ転じている
- 単にBBタッチやストキャス80越えのみを根拠とした逆張り提案は禁止する。
- ユーザーのトレードスタイル（スキャル／デイトレ）に応じて、戦略の時間軸を必ず適合させること。
- 初心者が再現可能なように、価格帯や視覚的特徴（ローソクの形・高値安値）を具体的に記述せよ。

【初心者への補足】
- インジケーターの基本的な解説（MACD、ストキャスなど）を簡潔に記述せよ。
- 感情に左右されず、損切りルールを厳守するよう助言せよ。
""")

# --- 共通処理の関数化（DRY原則） ---
def analyze_strategy(ticker, style, uploaded_files):
    messages_content = []
    filled_prompt = prompt_template.format_messages(ticker=ticker, style=style)
    messages_content.append({"type": "text", "text": filled_prompt[0].content})

    # 画像プレビューとbase64変換
    for uploaded_file in uploaded_files[:3]:
        st.image(uploaded_file, caption=uploaded_file.name, use_container_width=True)  # ←ここを修正
        image_bytes = uploaded_file.read()
        image_base64 = base64.b64encode(image_bytes).decode("utf-8")
        image_prompt = f"data:image/png;base64,{image_base64}"
        messages_content.append({"type": "image_url", "image_url": {"url": image_prompt}})

    messages = [HumanMessage(content=messages_content)]

    with st.spinner("AIがチャートを分析しています..."):
        response = llm(messages)
        st.markdown("### ✅ 分析結果")
        st.markdown(response.content)
        return response.content

def show_broker_section():
    st.markdown("""
---
#### 💡 この戦略を試すには、FX口座が必要です
AIが選ぶ、信頼と実績のある海外FX業者を比較しました。あなたに合った口座を選びましょう！
""")
    st.markdown("""
| 業者名 | 特徴 | スプレッド | レバレッジ | ボーナス | おすすめユーザー |
|--------|------|-------------|--------------|----------|------------------|
| **XM** | 信頼性・安定性◎ 日本語サポートあり | 普通（USD/JPY 1.6pips〜） | 最大1000倍 | あり（口座開設＋入金） | 安心して始めたい初心者向け |
| **BIG BOSS** | 約定スピード最速レベル 仮想通貨対応 | 狭め（ECN口座で0.1pips〜） | 最大999倍 | 入金ボーナス豊富 | スキャルパー・短期トレーダー |
| **AXIORY** | 手数料安く透明性◎ スプレッド狭い | 狭い（スタンダード口座で1.3pips〜） | 最大400倍 | 限定キャンペーンあり | コスト重視の中級者向け |
""")
    st.markdown("""
👇 あなたに合った業者を選んで、すぐに口座開設してみましょう！
""")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.link_button("✅ XMで口座開設", "https://www.xmtrading.com/jp/referral?token=1JpTZY4dh4n8F2lMoiJUkQ")
    with col2:
        st.link_button("🚀 BIG BOSSに登録", "https://www.bigboss-financial.com/ja?aid=AXKlzfNA")
    with col3:
        st.link_button("💼 AXIORYをチェック", "https://go.axiory.com/afs/come.php?cid=2077&ctgid=1043&atype=1&brandid=3")

# --- UIページ切替 ---
st.sidebar.title("📊 メニュー")
page = st.sidebar.radio("ページ選択", ("戦略提案", "保存済み戦略", "トレードカレンダー"))

if page == "戦略提案":
    st.title("📈 AI Entry Lab")
    with st.form("input_form_strategy"):
        ticker_input = st.text_input("通貨ペア／銘柄名", placeholder="例: USD/JPY, S&P500")
        style_input = st.selectbox("あなたのトレードスタイルは？", ["スキャルピング", "デイトレード"])
        files_input = st.file_uploader("チャート画像を最大3枚アップロード（JPG/PNG）", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        submit_input = st.form_submit_button("戦略を提案する")

    if submit_input:
        if files_input:
            result = analyze_strategy(ticker_input, style_input, files_input)
            today = datetime.now().strftime("%Y-%m-%d")
            st.session_state.strategy_history[today] = {
                "ticker": ticker_input,
                "style": style_input,
                "result": result
            }
            show_broker_section()
        else:
            st.warning("チャート画像をアップロードしてください。")

elif page == "保存済み戦略":
    st.title("📂 過去の戦略履歴")
    if st.session_state.strategy_history:
        selected_date = st.selectbox("日付を選んで戦略を確認", list(st.session_state.strategy_history.keys())[::-1])
        strategy = st.session_state.strategy_history[selected_date]
        st.markdown(f"#### 通貨／銘柄: {strategy['ticker']} ({strategy['style']})")
        st.markdown(strategy['result'])
        # 削除機能
        if st.button("この戦略を削除"):
            del st.session_state.strategy_history[selected_date]
            st.success("削除しました")
            st.experimental_rerun()
    else:
        st.info("まだ戦略は保存されていません。")

elif page == "トレードカレンダー":
    st.title("🗓️ トレードプラン実行メモ")
    today = datetime.now().strftime("%Y-%m-%d")
    note = st.text_area("本日の気づき／振り返りを記録", st.session_state.calendar_notes.get(today, ""))
    if st.button("保存"):
        st.session_state.calendar_notes[today] = note
        st.success("保存しました")

    st.markdown("---")
    st.markdown("### 📅 過去の記録")
    for date in sorted(st.session_state.calendar_notes.keys(), reverse=True):
        st.markdown(f"**{date}**: {st.session_state.calendar_notes[date]}")
