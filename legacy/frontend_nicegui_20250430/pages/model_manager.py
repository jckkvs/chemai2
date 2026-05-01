"""
Model Manager UI (Hugging Face / Local)
"""
import os
import asyncio
from pathlib import Path
from nicegui import ui

try:
    from huggingface_hub import login, snapshot_download
    _HAS_HF_HUB = True
except ImportError:
    _HAS_HF_HUB = False


def render_model_manager():
    """モデル管理（ダウンロード）タブの描画"""
    with ui.card().classes("w-full q-pa-md"):
        ui.label("🤗 Hugging Face モデル重みダウンロード").classes("text-lg font-bold hero-gradient q-mb-md")

        if not _HAS_HF_HUB:
            ui.label("⚠️ huggingface-hub がインストールされていません。pip install huggingface-hub を実行してください。").classes("text-red")
            return

        ui.label("ChemAIや各種深層学習・表現学習モデルに必要な事前学習重みをダウンロードします。").classes("text-caption text-grey-5 q-mb-lg")

        # トークン
        hf_token = ui.input("Access Token", password=True, placeholder="hf_...").props("outlined dense").classes("w-full q-mb-md")
        hf_token.tooltip("Hugging FaceのSettings画面で発行したRead権限トークンを入力（公開リポジトリなら不要な場合もあります）")

        # ターゲットリポジトリ選択
        model_repo = ui.select(
            options={
                "jckkvs/molai-chem-v1": "MolAI 化学構造エンコーダ (v1)",
                "jckkvs/unipka-base": "UniPKa 物性予測モデル",
                "custom": "カスタムリポジトリ..."
            },
            label="ダウンロード対象モデル",
            value="jckkvs/molai-chem-v1"
        ).props("outlined dense").classes("w-full q-mb-sm")

        custom_repo = ui.input("カスタムリポジトリID", placeholder="user/repo-name").props("outlined dense").classes("w-full q-mb-md")
        custom_repo.bind_visibility_from(model_repo, "value", value="custom")

        # プロキシ・詳細設定
        with ui.expansion("🔧 詳細設定（プロキシ等）", icon="settings").classes("w-full q-mb-md glass-card"):
            proxy_url = ui.input("Proxy URL", placeholder="http://proxy.example.com:8080").props("outlined dense").classes("w-full q-mb-sm")
            no_proxy = ui.input("No-Proxy", placeholder="localhost,127.0.0.1").props("outlined dense").classes("w-full")

        # ログコンテナ
        log_view = ui.log(max_lines=30).classes("w-full h-32 q-mb-md").style("font-size: 0.8rem; background: rgba(0,0,0,0.5);")

        # ダウンロード非同期処理
        async def on_download():
            repo_id = custom_repo.value if model_repo.value == "custom" else model_repo.value
            if not repo_id:
                ui.notify("⚠️ 対象リポジトリを選択または入力してください", type="warning")
                return

            btn_download.disable()
            log_view.clear()
            log_view.push(f"⏳ ダウンロード開始: {repo_id}")

            try:
                # 環境変数の設定
                if hf_token.value:
                    os.environ["HF_TOKEN"] = hf_token.value
                    # 同期関数なのでスレッド等で包むのが本来推奨されるがここでは簡易に
                    login(token=hf_token.value, add_to_git_credential=False)
                    log_view.push("✅ トークン認証完了")

                if proxy_url.value:
                    os.environ["HTTP_PROXY"] = proxy_url.value
                    os.environ["HTTPS_PROXY"] = proxy_url.value
                if no_proxy.value:
                    os.environ["NO_PROXY"] = no_proxy.value

                # 実際のダウンロード (IO-boundのため非同期にラップ)
                from huggingface_hub import snapshot_download
                local_dir = Path("models") / repo_id.split("/")[-1]
                local_dir.mkdir(parents=True, exist_ok=True)

                def _download_task():
                    return snapshot_download(
                        repo_id=repo_id,
                        local_dir=str(local_dir),
                        local_dir_use_symlinks=False,
                        resume_download=True
                    )

                log_view.push("📥 ファイルをフェッチ中... (コンソールに詳細が表示されます)")
                await asyncio.get_event_loop().run_in_executor(None, _download_task)

                log_view.push(f"🎉 ダウンロード成功！ 保存先: {local_dir.absolute()}")
                ui.notify(f"✅ {repo_id} のダウンロードが完了しました", type="positive")

            except Exception as e:
                log_view.push(f"❌ エラー発生:\n{str(e)}")
                ui.notify(f"❌ エラー: {str(e)}", type="negative")
            finally:
                btn_download.enable()

        btn_download = ui.button("📥 ダウンロード開始", on_click=on_download).classes("btn-primary w-full").props("size=md icon=cloud_download")
