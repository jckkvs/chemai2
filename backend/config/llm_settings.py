"""
backend/config/llm_settings.py
LLM API設定管理
- 設定は非表示のダイアログからアクセス可能
- 環境変数/設定ファイル/手動入力の3段階で設定
"""
import os
import json
from pathlib import Path
from typing import Optional, Dict, Any
from dataclasses import dataclass, asdict, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class LLMConfig:
    """LLM設定のデータクラス"""
    # モード: 'local' | 'api' | 'prompt_only'
    mode: str = 'prompt_only'
    
    # API設定（mode='api'時）
    api_endpoint: Optional[str] = None
    api_key: Optional[str] = None  # 環境変数推奨
    model_name: str = 'gpt-4o-mini'
    temperature: float = 0.1
    max_tokens: int = 2000
    
    # ローカルLLM設定（mode='local'時）
    local_model_path: Optional[str] = None
    local_device: str = 'cuda'  # 'cuda' | 'cpu'
    local_max_length: int = 2048
    
    # 動作設定
    enable_code_execution: bool = False  # LLM生成コードの自動実行（セキュリティ注意）
    sandbox_mode: bool = True  # コード実行をサンドボックスで制限
    log_prompts: bool = False  # プロンプトログの記録
    
    # UI設定
    show_advanced_options: bool = False  # 詳細設定を表示
    auto_save: bool = True  # 設定変更を自動保存
    
    # メタ情報（設定用、出力しない）
    _config_path: Optional[str] = field(default=None, repr=False)
    
    def to_dict(self) -> Dict[str, Any]:
        """JSONシリアライズ用（_で始まるキーは除外）"""
        return {k: v for k, v in asdict(self).items() if not k.startswith('_')}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], config_path: Optional[str] = None) -> 'LLMConfig':
        """辞書からインスタンス生成"""
        data = data.copy()
        config_path = data.pop('_config_path', config_path)
        # Filter out keys not in dataclass fields
        import dataclasses
        valid_keys = {f.name for f in dataclasses.fields(cls)}
        filtered_data = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered_data, _config_path=config_path)
    
    def save(self, path: Optional[str] = None):
        """設定をファイルに保存"""
        save_path = path or self._config_path or 'config/llm_settings.json'
        # Ensure the directory exists relative to project root or absolute path
        p = Path(save_path)
        if not p.is_absolute():
            # If it's something like 'config/llm_settings.json', it might be relative to the script
            # For chemai2, we usually want it in the project root's config dir
            pass
        p.parent.mkdir(parents=True, exist_ok=True)
        
        with open(p, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info(f"LLM設定を保存: {save_path}")
    
    @classmethod
    def load(cls, path: str = 'config/llm_settings.json') -> 'LLMConfig':
        """ファイルから設定を読み込み"""
        if not Path(path).exists():
            logger.info(f"設定ファイルが見つかりません: {path}。デフォルト設定を使用します。")
            return cls(_config_path=path)
        
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            return cls.from_dict(data, config_path=path)
        except Exception as e:
            logger.warning(f"設定ファイルの読み込みに失敗しました: {e}。デフォルト設定を使用します。")
            return cls(_config_path=path)
    
    @classmethod
    def from_env(cls) -> 'LLMConfig':
        """環境変数から設定を読み込み（優先度高）"""
        config = cls()
        
        # 環境変数マッピング
        env_mapping = {
            'CHEMAI_LLM_MODE': 'mode',
            'CHEMAI_LLM_API_ENDPOINT': 'api_endpoint',
            'CHEMAI_LLM_API_KEY': 'api_key',  # 注意: 環境変数経由で設定推奨
            'CHEMAI_LLM_MODEL_NAME': 'model_name',
            'CHEMAI_LLM_TEMPERATURE': 'temperature',
            'CHEMAI_LLM_LOCAL_PATH': 'local_model_path',
        }
        
        for env_var, attr in env_mapping.items():
            value = os.getenv(env_var)
            if value is not None:
                if attr in ['temperature']:
                    setattr(config, attr, float(value))
                else:
                    setattr(config, attr, value)
        
        return config
    
    def get_effective_config(self) -> 'LLMConfig':
        """
        有効な設定を解決（環境変数 > ファイル > デフォルト）
        """
        # 環境変数が最優先
        env_config = self.from_env()
        
        # 環境変数で設定された項目のみ上書き
        for key, value in env_config.to_dict().items():
            if value is not None and key != '_config_path':
                setattr(self, key, value)
        
        return self
