"""
frontend_streamlit/pages/pipeline/llm_assistant_page.py

LLM アシスタントページ：データ整形・解析方針のヒアリング支援

機能:
  - 整形されていないデータの LLM による自動修正
  - 外部 LLM 用プロンプト生成（ChatGPT, Claude 等）
  - データ解析方針のヒアリング支援
  - パワポ・ワード資料からの情報抽出（将来的に拡張）
"""
from __future__ import annotations

import logging
import pandas as pd
import streamlit as st

logger = logging.getLogger(__name__)


def render() -> None:
    """LLM アシスタントページのレンダリング。"""
    st.markdown("## 🤖 LLM データアシスタント")
    
    # セッション初期化
    if "llm_cleaning_code" not in st.session_state:
        st.session_state["llm_cleaning_code"] = ""
    if "external_prompt" not in st.session_state:
        st.session_state["external_prompt"] = ""
    if "cleaned_df" not in st.session_state:
        st.session_state["cleaned_df"] = None
    if "analysis_plan" not in st.session_state:
        st.session_state["analysis_plan"] = {}
    
    df = st.session_state.get("df")
    
    if df is None:
        st.info("📂 まずデータをアップロードしてください")
        if st.button("📂 データ読み込みページへ"):
            st.session_state["page"] = "data_load"
            st.rerun()
        return
    
    # タブ構成
    tab1, tab2, tab3 = st.tabs([
        "🧹 データ整形 (LLM)",
        "📋 解析方針ヒアリング",
        "🔗 外部 LLM プロンプト",
    ])
    
    with tab1:
        _render_data_cleaning_tab(df)
    
    with tab2:
        _render_analysis_planning_tab(df)
    
    with tab3:
        _render_external_llm_tab(df)


def _render_data_cleaning_tab(df: pd.DataFrame) -> None:
    """データ整形タブのレンダリング。"""
    st.markdown("### 🧹 LLM によるデータ整形")
    
    st.markdown("""
**LLM があなたのデータを分析し、クリーニング用の Python コードを自動生成します。**
    
対応する問題例:
- セル結合の解消
- 列名の誤字・空白の修正
- 欠損値の適切な処理
- データ型の混在解消
- 不要なヘッダー行・フッター行の削除
    """)
    
    # データ品質分析の実行
    if st.button("🔍 データ品質を分析", use_container_width=True):
        with st.spinner("データを分析中..."):
            from backend.data.llm_data_cleaner import analyze_data_quality
            
            report = analyze_data_quality(df)
            st.session_state["quality_report"] = report
            
            if report.is_clean:
                st.success("✅ データはきれいに整形されています！")
            else:
                st.warning(f"⚠️ {len(report.issues)} 個の問題を検出しました")
        
        st.rerun()
    
    # 品質レポートの表示
    report = st.session_state.get("quality_report")
    if report:
        _display_quality_report(report)
        
        # LLM コード生成
        if report.needs_cleaning:
            st.markdown("---")
            st.markdown("### 🔧 クリーニングコードの生成")
            
            col1, col2 = st.columns(2)
            with col1:
                provider = st.selectbox(
                    "LLM プロバイダー",
                    ["stub", "openai"],
                    help="stub はテスト用、openai は API キーが必要"
                )
            
            with col2:
                use_external = st.checkbox(
                    "外部 LLM 用プロンプトを生成",
                    help="ChatGPT や Claude などに手動で貼り付けて使用"
                )
            
            if st.button("🤖 クリーニングコードを生成", use_container_width=True):
                with st.spinner("LLM がコードを生成中..."):
                    from backend.data.llm_data_cleaner import generate_cleaning_code
                    
                    code, external_prompt = generate_cleaning_code(
                        df, report, 
                        provider_name=provider,
                        use_external_llm=use_external
                    )
                    
                    st.session_state["llm_cleaning_code"] = code
                    st.session_state["external_prompt"] = external_prompt
                
                st.rerun()
        
        # 生成されたコードの表示と実行
        if st.session_state["llm_cleaning_code"]:
            st.markdown("---")
            st.markdown("### 📝 生成されたクリーニングコード")
            
            st.code(st.session_state["llm_cleaning_code"], language="python")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("▶️ コードを実行してクリーニング", use_container_width=True):
                    with st.spinner("クリーニング実行中..."):
                        from backend.data.llm_data_cleaner import execute_cleaning_code
                        
                        cleaned_df, log_msg = execute_cleaning_code(
                            st.session_state["llm_cleaning_code"],
                            df
                        )
                        
                        st.session_state["cleaned_df"] = cleaned_df
                        st.session_state["cleaning_log"] = log_msg
                    
                    st.rerun()
            
            with c2:
                if st.button("📋 コードをコピー", use_container_width=True):
                    st.code(st.session_state["llm_cleaning_code"], language="python")
                    st.success("コードが表示されました。コピーしてご利用ください。")
        
        # クリーニング結果の表示
        if st.session_state["cleaned_df"] is not None:
            st.markdown("---")
            st.markdown("### ✅ クリーニング結果")
            
            log = st.session_state.get("cleaning_log", "")
            if log:
                st.info(log)
            
            cleaned_df = st.session_state["cleaned_df"]
            
            # 比較表示
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**元のデータ**")
                st.write(f"形状：{df.shape}")
                st.dataframe(df.head(), use_container_width=True)
            
            with col2:
                st.markdown("**クリーニング後**")
                st.write(f"形状：{cleaned_df.shape}")
                st.dataframe(cleaned_df.head(), use_container_width=True)
            
            # 適用ボタン
            if st.button("✨ このデータを適用して解析を続ける", type="primary"):
                st.session_state["df"] = cleaned_df
                st.session_state["file_name"] = f"{st.session_state.get('file_name', 'data')}_cleaned"
                st.session_state["cleaned_df"] = None
                st.session_state["llm_cleaning_code"] = ""
                st.success("クリーニング済みのデータを適用しました！")
                st.rerun()


def _display_quality_report(report) -> None:
    """品質レポートを表示。"""
    st.markdown("#### 📊 データ品質レポート")
    
    if report.is_clean:
        st.success("✅ データはきれいに整形されています")
    else:
        st.warning(f"⚠️ データに問題があります（{len(report.issues)} 件）")
    
    if report.issues:
        st.markdown("**検出された問題:**")
        for issue in report.issues:
            st.markdown(f"- {issue}")
    
    if report.structure_issues:
        st.markdown("**構造的な問題:**")
        for issue in report.structure_issues:
            st.markdown(f"- {issue}")
    
    if report.suggestions:
        st.markdown("**提案事項:**")
        for suggestion in report.suggestions:
            st.markdown(f"- {suggestion}")


def _render_analysis_planning_tab(df: pd.DataFrame) -> None:
    """解析方針ヒアリングタブのレンダリング。"""
    st.markdown("### 📋 解析方針ヒアリング")
    
    st.markdown("""
**LLM があなたの研究目的をヒアリングし、最適な解析方針を提案します。**
    
以下の質問にお答えください:
    """)
    
    # ヒアリングフォーム
    with st.form("analysis_plan_form"):
        research_goal = st.text_area(
            "🎯 研究目的・達成したいことを教えてください",
            placeholder="例：新しい触媒の活性を予測したい、溶解度の高い化合物を見つけたい..."
        )
        
        data_description = st.text_area(
            "📊 データの説明",
            placeholder="例：実験で得られた収率データ、文献から収集した物性値..."
        )
        
        target_col = st.selectbox(
            "🎯 目的変数（予測したい値）",
            options=df.columns.tolist(),
            help="回帰分析の場合は連続値、分類分析の場合はカテゴリを選択"
        )
        
        constraints = st.text_area(
            "⚠️ 制約条件・注意点（任意）",
            placeholder="例：特定の記述子を使いたい、計算時間を短くしたい..."
        )
        
        submitted = st.form_submit_button("💡 解析方針を提案してもらう", use_container_width=True)
        
        if submitted:
            if not research_goal.strip():
                st.error("研究目的を入力してください")
            else:
                with st.spinner("解析方針を生成中..."):
                    plan = _generate_analysis_plan(
                        df, research_goal, data_description, 
                        target_col, constraints
                    )
                    st.session_state["analysis_plan"] = plan
                
                st.rerun()
    
    # 生成された解析方針の表示
    plan = st.session_state.get("analysis_plan", {})
    if plan:
        st.markdown("---")
        st.markdown("### 📝 提案された解析方針")
        
        if "plan_text" in plan:
            st.markdown(plan["plan_text"])
        
        if "recommended_steps" in plan:
            st.markdown("**推奨ステップ:**")
            for i, step in enumerate(plan["recommended_steps"], 1):
                st.markdown(f"{i}. {step}")
        
        if "suggested_models" in plan:
            st.markdown("**推奨モデル:**")
            for model in plan["suggested_models"]:
                st.markdown(f"- {model}")
        
        if "suggested_descriptors" in plan:
            st.markdown("**推奨記述子:**")
            for desc in plan["suggested_descriptors"]:
                st.markdown(f"- {desc}")
        
        # 方針を適用
        if st.button("🚀 この方針で解析を開始", type="primary"):
            st.session_state["target_col"] = target_col
            st.session_state["page"] = "automl"
            st.success("解析方針を適用しました。AutoML ページへ移動します。")
            st.rerun()


def _generate_analysis_plan(
    df: pd.DataFrame,
    goal: str,
    description: str,
    target: str,
    constraints: str
) -> dict:
    """解析方針を生成（現在はルールベース、将来的に LLM 化）。"""
    from backend.data.type_detector import TypeDetector
    
    detector = TypeDetector()
    detection = detector.detect(df)
    
    # タスク種の自動判定
    target_dtype = df[target].dtype
    is_classification = df[target].nunique() < 10 or str(target_dtype) == 'object'
    
    plan = {
        "goal": goal,
        "target": target,
        "task_type": "classification" if is_classification else "regression",
        "data_shape": df.shape,
        "n_features": len([c for c in df.columns if c != target]),
    }
    
    # シンプルなルールベースの提案
    steps = [
        "1. データの前処理（欠損値補完、外れ値除去）",
        "2. 特徴量の選択・次元削減",
    ]
    
    if is_classification:
        steps.append("3. 分類モデルの訓練（Random Forest, XGBoost, LightGBM）")
        steps.append("4. 交差検証による評価（accuracy, F1-score）")
        steps.append("5. 特徴量重要度の解釈（SHAP 値）")
        suggested_models = ["Random Forest", "XGBoost", "LightGBM", "SVM"]
    else:
        steps.append("3. 回帰モデルの訓練（Random Forest, XGBoost, GPR）")
        steps.append("4. 交差検証による評価（R², RMSE, MAE）")
        steps.append("5. 残差分析と特徴量重要度の解釈")
        suggested_models = ["Random Forest", "XGBoost", "LightGBM", "Gaussian Process"]
    
    plan["recommended_steps"] = steps
    plan["suggested_models"] = suggested_models
    plan["suggested_descriptors"] = ["RDKit 記述子", "Mordred 記述子", "MACCS キー"]
    
    # プレーンテキストの計画書
    plan_text = f"""## 解析方針

**研究目的:** {goal}

**目的変数:** `{target}` ({'分類' if is_classification else '回帰'}タスク)

**データ規模:** {df.shape[0]}サンプル × {df.shape[1]}特徴量

### 推奨アプローチ

このデータセットに対して、以下の手順で解析を行うことを推奨します:

{'\n'.join(steps)}

### 推奨モデル

- {' / '.join(suggested_models)}

### 次のステップ

「🚀 この方針で解析を開始」ボタンをクリックすると、AutoML ページに移動し、
自動的に最適なモデルの探索を開始します。
"""
    
    plan["plan_text"] = plan_text
    
    return plan


def _render_external_llm_tab(df: pd.DataFrame) -> None:
    """外部 LLM 用プロンプトタブのレンダリング。"""
    st.markdown("### 🔗 外部 LLM 用プロンプト")
    
    st.markdown("""
**高精度な LLM（ChatGPT-4, Claude 3, Gemini 等）を使用する場合のプロンプトを生成します。**
    
ローカル環境に高性能 LLM がデプロイされていない場合や、
より高精度な解析が必要な場合に、以下のプロンプトをコピーして
外部の LLM サービスにご利用ください。
    """)
    
    # プロンプト生成
    if st.button("📝 外部 LLM 用プロンプトを生成", use_container_width=True):
        from backend.data.llm_data_cleaner import analyze_data_quality, generate_cleaning_code
        
        report = analyze_data_quality(df)
        _, external_prompt = generate_cleaning_code(df, report, use_external_llm=True)
        
        st.session_state["external_prompt"] = external_prompt
    
    if st.session_state["external_prompt"]:
        st.markdown("#### 生成されたプロンプト")
        st.text_area(
            "プロンプト（コピーして ChatGPT 等にご利用ください）",
            value=st.session_state["external_prompt"],
            height=400,
            key="external_prompt_display"
        )
        
        if st.button("📋 プロンプトをコピー"):
            st.success("プロンプトが表示されました。クリップボードにコピーしてください。")
    
    # API 設定へのリンク（目立たない場所）
    st.markdown("---")
    with st.expander("⚙️ LLM API 設定（上級者向け）"):
        st.markdown("""
**LLM API の設定**

OpenAI API を使用する場合は、以下のように環境変数を設定してください:

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
```

または、Streamlit のシークレット管理を使用:

`.streamlit/secrets.toml` に以下を追加:
```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```
        """)
