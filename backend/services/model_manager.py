# backend/services/model_manager.py
import os
import asyncio
import logging
from typing import Optional, Callable
from huggingface_hub import snapshot_download

logger = logging.getLogger(__name__)

class ModelManager:
    """LLMモデルのローカル存在確認・初回ダウンロード・キャッシュ管理を担当するマネージャクラス。"""
    
    def __init__(self, model_id: str, local_dir: Optional[str] = None):
        self.model_id = model_id
        self.local_dir = local_dir or os.path.join(".", "models", os.path.basename(model_id))
        self._is_downloaded = self._check_local_exists()

    def _check_local_exists(self) -> bool:
        """モデルの主要ファイル（config.json）が存在し、サイズが0でないか確認。"""
        config_path = os.path.join(self.local_dir, "config.json")
        return os.path.exists(config_path) and os.path.getsize(config_path) > 0

    async def ensure_downloaded(self, progress_callback: Optional[Callable] = None) -> str:
        """モデルの存在を確認し、なければダウンロードを実行する。"""
        if self._is_downloaded:
            logger.info(f"モデルは既にローカルに存在します: {self.local_dir}")
            return self.local_dir

        logger.info(f"モデルダウンロード開始: {self.model_id} → {self.local_dir}")
        loop = asyncio.get_event_loop()
        
        def _download():
            try:
                # ネットワーク障害時はここから例外が発生する
                return snapshot_download(
                    repo_id=self.model_id,
                    local_dir=self.local_dir,
                    local_dir_use_symlinks=False,
                    resume_download=True,
                    force_download=False
                )
            except Exception as e:
                logger.error(f"モデルダウンロード失敗: {e}")
                raise

        try:
            path = await loop.run_in_executor(None, _download)
            self._is_downloaded = True
            logger.info(f"モデルダウンロード完了: {path}")
            return path
        except Exception as e:
            if progress_callback:
                # エラーメッセージをコールバックでUIに通知可能にする
                progress_callback(0.0, f"ダウンロード失敗: {str(e)}")
            raise
