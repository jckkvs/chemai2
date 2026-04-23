# tests/test_model_manager.py
import pytest
import os
from unittest.mock import MagicMock, patch
from backend.services.model_manager import ModelManager

def test_model_manager_init():
    manager = ModelManager("jckkvs/bonsai-8b-1.58bit", "./test_models")
    assert manager.model_id == "jckkvs/bonsai-8b-1.58bit"
    assert manager.local_dir == "./test_models"

def test_check_local_exists_not_found(tmp_path):
    local_dir = tmp_path / "model"
    manager = ModelManager("repo/id", str(local_dir))
    assert manager._check_local_exists() is False

def test_check_local_exists_found(tmp_path):
    local_dir = tmp_path / "model"
    local_dir.mkdir()
    config_file = local_dir / "config.json"
    config_file.write_text("{}")
    
    manager = ModelManager("repo/id", str(local_dir))
    assert manager._check_local_exists() is True

@pytest.mark.asyncio
async def test_ensure_downloaded_already_exists(tmp_path):
    local_dir = tmp_path / "model"
    local_dir.mkdir()
    (local_dir / "config.json").write_text("{}")
    
    manager = ModelManager("repo/id", str(local_dir))
    path = await manager.ensure_downloaded()
    assert path == str(local_dir)

@pytest.mark.asyncio
async def test_ensure_downloaded_trigger_download(tmp_path):
    local_dir = tmp_path / "model"
    manager = ModelManager("repo/id", str(local_dir))
    
    with patch("backend.services.model_manager.snapshot_download") as mock_download:
        mock_download.return_value = str(local_dir)
        # 擬似的にファイルを作成
        local_dir.mkdir(parents=True, exist_ok=True)
        (local_dir / "config.json").write_text("{}")
        
        path = await manager.ensure_downloaded()
        
        mock_download.assert_called_once_with(
            repo_id="repo/id",
            local_dir=str(local_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
            force_download=False
        )
        assert path == str(local_dir)
        assert manager._is_downloaded is True
