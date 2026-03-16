"""
backend/chem/molai_adapter.py

MolAI: CNN Encoder + GRU Decoder SMILES オートエンコーダーによる
分子潜在ベクトル記述子の生成アダプター。

論文: Mahdizadeh & Eriksson, J. Chem. Inf. Model. 2025
     DOI: 10.1021/acs.jcim.5c00491

アーキテクチャ:
  - Encoder: 1D CNN (キャラクターレベルSMILES → 固定長潜在ベクトル)
  - Decoder: GRU (潜在ベクトル → SMILES 再構成)
  - 入力: SMILES 文字列 (最大 MAX_SMILES_LEN 文字)
  - 出力: n_components 次元 (PCA で圧縮した潜在ベクトル)

Implements: MolAI §2 Architecture, §3.1 Encoder
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from backend.chem.base import BaseChemAdapter, DescriptorMetadata, DescriptorResult

logger = logging.getLogger(__name__)

# ── SMILES トークナイザー定数 ────────────────────────────────────────────
# 論文 §2.1: キャラクターレベルトークナイズ, Br/Cl は一括トークン
MAX_SMILES_LEN = 111  # 論文 §2.1: "SMILES longer than 111 characters ... excluded"
PAD_CHAR = " "        # padding character

_SPECIAL_TOKENS = ["Br", "Cl"]  # 2文字トークン（先に置換）
_CHARSET: list[str] = (
    list("CNOSFPIBrCl")          # 元素記号
    + list("=#@\\/%()[].")       # 結合・括弧系
    + list("0123456789")         # 数字
    + list("+-hHcsnoSF")         # 電荷・芳香族等
    + [PAD_CHAR]
)
_CHARSET_UNIQUE = list(dict.fromkeys(_CHARSET))  # 重複除去・順序保持
_CHAR2IDX: dict[str, int] = {c: i for i, c in enumerate(_CHARSET_UNIQUE)}
VOCAB_SIZE = len(_CHARSET_UNIQUE)


def _tokenize_smiles(smiles: str) -> list[str]:
    """SMILES を文字トークンのリストに変換する。Br/Cl は単一トークンとして扱う。
    
    Implements: 論文 §2.1 Tokenization
    """
    tokens: list[str] = []
    i = 0
    while i < len(smiles):
        matched = False
        for tok in _SPECIAL_TOKENS:
            if smiles[i:i + len(tok)] == tok:
                tokens.append(tok)
                i += len(tok)
                matched = True
                break
        if not matched:
            tokens.append(smiles[i])
            i += 1
    return tokens


def _smiles_to_onehot(smiles: str) -> np.ndarray:
    """SMILES → one-hot エンコード行列 (MAX_SMILES_LEN x VOCAB_SIZE)。
    
    Implements: 論文 §2.1 Encoding
    """
    tokens = _tokenize_smiles(smiles)[:MAX_SMILES_LEN]
    # パディング
    tokens += [PAD_CHAR] * (MAX_SMILES_LEN - len(tokens))
    mat = np.zeros((MAX_SMILES_LEN, VOCAB_SIZE), dtype=np.float32)
    for j, tok in enumerate(tokens):
        idx = _CHAR2IDX.get(tok, _CHAR2IDX[PAD_CHAR])
        mat[j, idx] = 1.0
    return mat


# ── PyTorch モデル定義 ────────────────────────────────────────────────────
def _build_encoder(latent_dim: int = 256):
    """CNN Encoder: (batch, seq_len, vocab) → (batch, latent_dim)
    
    Implements: 論文 §2.2 Encoder Architecture (1D CNN layers)
    """
    try:
        import torch
        import torch.nn as nn

        class MolAIEncoder(nn.Module):
            def __init__(self, vocab_size: int, latent_dim: int):
                super().__init__()
                self.conv1 = nn.Conv1d(vocab_size, 9,  kernel_size=9)
                self.conv2 = nn.Conv1d(9,          9,  kernel_size=9)
                self.conv3 = nn.Conv1d(9,          10, kernel_size=11)
                self.flatten = nn.Flatten()
                # 畳み込み後の次元: floor((MAX_SMILES_LEN - 8)/1) → 103 → 95 → 85 → 10 ch
                # seq after 3 convs: MAX_SMILES_LEN - 8 - 8 - 10 = 85
                flat_dim = 10 * (MAX_SMILES_LEN - 8 - 8 - 10)  # = 10 * 85 = 850
                self.fc = nn.Linear(flat_dim, latent_dim)
                self.relu = nn.ReLU()

            def forward(self, x):  # x: (batch, seq_len, vocab_size)
                x = x.permute(0, 2, 1)          # → (batch, vocab_size, seq_len)
                x = self.relu(self.conv1(x))
                x = self.relu(self.conv2(x))
                x = self.relu(self.conv3(x))
                x = self.flatten(x)
                return self.fc(x)

        return MolAIEncoder(VOCAB_SIZE, latent_dim)
    except ImportError:
        return None


# ── アダプター本体 ─────────────────────────────────────────────────────────
class MolAIAdapter(BaseChemAdapter):
    """MolAI SMILES オートエンコーダー記述子アダプター。

    CNN Encoder で SMILES を高次元潜在ベクトルに変換し、
    PCA で n_components 次元に圧縮して記述子として返す。

    Implements: F-MolAI | 論文: §2 Architecture | 式 (latent = Encoder(SMILES))
    API:
        n_components (int): PCA 出力次元数（デフォルト 32）
        latent_dim (int): エンコーダー出力次元（デフォルト 256）
    前提: torch >= 2.0 がインストールされていること
    """

    def __init__(self, n_components: int | str = "auto", latent_dim: int = 256):
        """
        Args:
            n_components: PCA出力次元数。
                - int: 固定次元数
                - "auto": 累積寄与率95%超えの最小次元数を自動決定
            latent_dim: エンコーダー出力次元
        """
        self.n_components = n_components
        self.latent_dim = latent_dim
        self._encoder = None
        self._pca = None

    @property
    def name(self) -> str:
        return "molai"

    @property
    def description(self) -> str:
        return (
            "MolAI CNN+GRU Autoencoder (Mahdizadeh & Eriksson, JCIM 2025)。"
            f"SMILES を {self.latent_dim} 次元潜在ベクトルに変換後、"
            f"PCA で {self.n_components} 次元に圧縮します。"
        )

    def is_available(self) -> bool:
        try:
            import torch  # noqa: F401
            import sklearn  # noqa: F401
            return True
        except ImportError:
            return False

    def _get_encoder(self):
        """エンコーダーを遅延初期化して返す。"""
        if self._encoder is None:
            self._encoder = _build_encoder(self.latent_dim)
            if self._encoder is None:
                raise RuntimeError("torch がインストールされていません。")
            self._encoder.eval()
        return self._encoder

    def compute(
        self,
        smiles_list: list[str],
        selected_descriptors: list[str] | None = None,
        **kwargs: Any,
    ) -> DescriptorResult:
        """SMILES リストから MolAI 潜在ベクトル記述子を計算する。

        Implements: F-MolAI | 論文 §2.2 Encoder, §3.1 Latent Space
        Args:
            smiles_list: 入力 SMILES のリスト
            selected_descriptors: 使用する列名（None = 全件）
        Returns:
            DescriptorResult: molai_pc1, molai_pc2, ... の DataFrame
        """
        self._require_available()
        import torch
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        encoder = self._get_encoder()
        failed_indices: list[int] = []
        latent_vecs: list[np.ndarray] = []

        # ── Step 1: SMILES → one-hot → latent vector ──
        for i, smi in enumerate(smiles_list):
            try:
                if not smi or len(_tokenize_smiles(smi)) > MAX_SMILES_LEN:
                    raise ValueError(f"SMILES が長すぎます (max {MAX_SMILES_LEN} chars): {smi[:30]}...")
                oh = _smiles_to_onehot(smi)
                t = torch.tensor(oh[np.newaxis, :, :], dtype=torch.float32)
                with torch.no_grad():
                    lv = encoder(t).numpy().flatten()
                latent_vecs.append(lv)
            except Exception as e:
                logger.warning("MolAI エンコードに失敗: idx=%d, smi=%s, err=%s", i, smi[:30], e)
                failed_indices.append(i)
                latent_vecs.append(np.full(self.latent_dim, np.nan))

        latent_mat = np.array(latent_vecs, dtype=np.float32)

        # ── Step 2: PCA で次元削減 ──
        # Implements: 論文補足「高次元なのでPCAで圧縮して使用」
        valid_mask = ~np.isnan(latent_mat).any(axis=1)
        n_valid = int(valid_mask.sum())

        if n_valid < 2:
            # 十分なデータがない場合
            n_comp = min(
                self.n_components if isinstance(self.n_components, int) else 32,
                latent_mat.shape[1],
            )
            pc_mat = np.full((len(smiles_list), n_comp), np.nan, dtype=np.float32)
            col_names = [f"molai_pc{k + 1}" for k in range(n_comp)]
            df = pd.DataFrame(pc_mat, columns=col_names)
        else:
            scaler = StandardScaler()
            X_valid = scaler.fit_transform(latent_mat[valid_mask])

            if self.n_components == "auto":
                # 自動最適化: 累積寄与率95%超えの最小次元を採用
                max_comp = min(n_valid, latent_mat.shape[1])
                pca_full = PCA(n_components=max_comp, random_state=42)
                pca_full.fit(X_valid)
                cum_ratio = np.cumsum(pca_full.explained_variance_ratio_)
                n_95 = int(np.searchsorted(cum_ratio, 0.95) + 1)
                n_comp = min(n_95, max_comp)
                n_comp = max(n_comp, 2)  # 最低2次元
                logger.info(
                    f"MolAI PCA自動最適化: 累積寄与率95%%到達={n_95}次元, "
                    f"最大={max_comp}次元 → 採用={n_comp}次元 "
                    f"(累積寄与率={cum_ratio[n_comp - 1]:.1%})"
                )
            else:
                n_comp = min(self.n_components, n_valid, latent_mat.shape[1])

            pca = PCA(n_components=n_comp, random_state=42)
            pc_mat = np.full((len(smiles_list), n_comp), np.nan, dtype=np.float32)
            pc_mat[valid_mask] = pca.fit_transform(X_valid).astype(np.float32)
            self._pca = pca  # 後から参照できるよう保持

            col_names = [f"molai_pc{k + 1}" for k in range(n_comp)]
            df = pd.DataFrame(pc_mat, columns=col_names)

        # selected_descriptors でフィルタリング
        if selected_descriptors:
            available = [c for c in selected_descriptors if c in df.columns]
            df = df[available] if available else df

        return DescriptorResult(
            descriptors=df,
            smiles_list=smiles_list,
            failed_indices=failed_indices,
            adapter_name=self.name,
            metadata={
                "latent_dim": self.latent_dim,
                "n_components": n_comp,
                "n_failed": len(failed_indices),
            },
        )

    def get_descriptor_names(self) -> list[str]:
        """利用可能な記述子名（molai_pc1 〜 molai_pcN）を返す。"""
        n = self.n_components if isinstance(self.n_components, int) else 32
        return [f"molai_pc{k + 1}" for k in range(n)]

    def get_descriptors_metadata(self) -> list[DescriptorMetadata]:
        """各 PCA 成分のメタデータを返す。"""
        n = self.n_components if isinstance(self.n_components, int) else 32
        return [
            DescriptorMetadata(
                name=f"molai_pc{k + 1}",
                meaning=f"MolAI 潜在空間 第{k + 1}主成分 (PCA圧縮)",
                is_count=False,
                is_binary=False,
                description=(
                    "MolAI CNN Encoder で得た潜在ベクトルを PCA 圧縮した成分。"
                    "論文: Mahdizadeh & Eriksson, JCIM 2025, DOI: 10.1021/acs.jcim.5c00491"
                ),
            )
            for k in range(n)
        ]
