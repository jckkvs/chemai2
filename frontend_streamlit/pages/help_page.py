"""
frontend_streamlit/pages/help_page.py

目的変数に対する推奨説明変数のヘルプビュー画面
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from backend.chem.recommender import get_all_target_recommendations

def render_help_page() -> None:
    """推奨説明変数のヘルプ・データベース一覧ページ"""
    st.markdown("## 📚 物理化学物性: 推奨説明変数データベース")
    st.markdown("""
        各目的変数（予測したい物理化学的な性質）に対して、事前に有効性が知られている推奨説明変数（記述子）の一覧です。
        モデルの精度を高め、結果の解釈性を保つためには、やみくもに全変数を投入するのではなく、
        **目的にあった適切な変数を少数（8〜20個程度）選定することが最良のプラクティス**となります。
    """)

    st.markdown("---")

    recommendations = get_all_target_recommendations()

    for idx, rec in enumerate(recommendations):
        # 目的変数ごとに展開可能なExpander（デフォルトは閉じる）
        with st.expander(f"🔹 {rec.target_name}", expanded=False):
            st.markdown(f"**【事前知識】**\n{rec.summary}")
            
            # 記述子のリストをDataFrameに変換してリッチにテーブル表示
            desc_data = []
            for d in rec.descriptors:
                desc_data.append({
                    "カテゴリ": d.library,
                    "推奨記述子名": d.name,
                    "物理的・化学的意味": d.meaning,
                    "根拠ソース・主な論文": d.source,
                })
            df_desc = pd.DataFrame(desc_data)
            
            # Streamlitのdataframe表示でスタイリング
            st.dataframe(
                df_desc,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "カテゴリ": st.column_config.TextColumn("ライブラリ", width="small"),
                    "推奨記述子名": st.column_config.TextColumn("記述子名", width="medium"),
                    "物理的・化学的意味": st.column_config.TextColumn("意味付け", width="large"),
                    "根拠ソース・主な論文": st.column_config.TextColumn("出典", width="medium"),
                }
            )

if __name__ == "__main__":
    # 単独実行テスト用
    st.set_page_config(page_title="推奨変数ヘルプ", layout="wide")
    render_help_page()
