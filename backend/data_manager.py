"""
Data Management Module - chemai2/backend/data_manager.py
Handles data upload, validation, metadata extraction, and SMILES processing
"""
import hashlib
import json
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Union, Literal

import pandas as pd
from pydantic import BaseModel, Field, field_validator
from fastapi import UploadFile, HTTPException

from backend.core.config import settings
from backend.utils.logger import logger
from backend.chem.smiles_utils import standardize_smiles, validate_smiles_batch


class DataColumnMeta(BaseModel):
    """Metadata for a single data column"""
    name: str
    dtype: Literal['numeric', 'categorical', 'binary', 'smiles', 'text', 'datetime']
    nullable: bool
    missing_count: int
    unique_count: int
    sample_values: List[str] = Field(default_factory=list)
    statistics: Optional[Dict[str, float]] = None  # mean, std, min, max for numeric
    
    @field_validator('sample_values')
    @classmethod
    def limit_samples(cls, v):
        return v[:5]  # Limit to 5 samples for performance


class DatasetMeta(BaseModel):
    """Metadata for uploaded dataset"""
    dataset_id: str
    filename: str
    nrows: int
    ncols: int
    file_hash: str  # SHA256 for deduplication
    columns: List[DataColumnMeta]
    smiles_column: Optional[str] = None
    target_column: Optional[str] = None
    created_at: str
    tags: List[str] = Field(default_factory=list)


class DataManager:
    """Centralized data management with validation and caching"""
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path or settings.DATA_DIR
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._metadata_cache: Dict[str, DatasetMeta] = {}
    
    async def upload_dataset(
        self,
        file: UploadFile,
        user_id: str,
        detect_smiles: bool = True,
        auto_standardize: bool = True
    ) -> DatasetMeta:
        """
        Upload and validate dataset with automatic metadata extraction
        """
        # Read file with format detection
        df = await self._read_file(file)
        
        # Generate unique ID and hash
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        dataset_id = f"{user_id}_{hashlib.md5(file_content[:1024]).hexdigest()[:12]}"
        
        # Check for duplicate upload
        if self._check_duplicate(file_hash):
            logger.info(f"Duplicate dataset detected: {file_hash}")
            return self._metadata_cache[file_hash]
        
        # Extract column metadata
        columns_meta = []
        smiles_candidates = []
        
        for col in df.columns:
            col_meta = await self._analyze_column(df[col], col_name=col)
            columns_meta.append(col_meta)
            
            if col_meta.dtype == 'smiles' or (detect_smiles and self._is_smiles_candidate(df[col])):
                smiles_candidates.append(col)
        
        # Auto-select or validate SMILES column
        smiles_column = None
        if smiles_candidates and auto_standardize:
            selected_col = smiles_candidates[0]
            standardized = standardize_smiles(df[selected_col].dropna().tolist())
            if standardized:
                smiles_column = selected_col
                # To maintain parity with original length, we handle NaNs
                full_std = [None] * len(df)
                idx = 0
                for i, is_na in enumerate(df[selected_col].isna()):
                    if not is_na:
                        full_std[i] = standardized[idx]
                        idx += 1
                df[f"{selected_col}_std"] = full_std
        
        # Save dataset and metadata
        file_path = self.storage_path / f"{dataset_id}.parquet"
        df.to_parquet(file_path, index=False)
        
        meta = DatasetMeta(
            dataset_id=dataset_id,
            filename=file.filename,
            nrows=len(df),
            ncols=len(df.columns),
            file_hash=file_hash,
            columns=columns_meta,
            smiles_column=smiles_column,
            created_at=pd.Timestamp.now().isoformat()
        )
        
        # Cache metadata
        self._metadata_cache[dataset_id] = meta
        self._metadata_cache[file_hash] = meta
        
        # Save metadata to disk
        meta_path = self.storage_path / f"{dataset_id}_meta.json"
        with open(meta_path, 'w', encoding='utf-8') as f:
            f.write(meta.model_dump_json(indent=2))
        
        logger.info(f"Dataset uploaded: {dataset_id} ({len(df)} rows, {len(df.columns)} cols)")
        return meta
    
    async def _analyze_column(self, series: pd.Series, col_name: str) -> DataColumnMeta:
        """Analyze single column to determine type and statistics"""
        non_null = series.dropna()
        
        # Type detection logic
        if non_null.empty:
            dtype = 'text'
        elif self._is_smiles_series(non_null):
            dtype = 'smiles'
        elif pd.api.types.is_bool_dtype(non_null) or set(non_null.unique()) <= {0, 1, True, False}:
            dtype = 'binary'
        elif pd.api.types.is_categorical_dtype(non_null) or (non_null.nunique() / (len(non_null) + 1) < 0.05):
            dtype = 'categorical'
        elif pd.api.types.is_numeric_dtype(non_null):
            dtype = 'numeric'
        else:
            dtype = 'text'
        
        # Statistics calculation
        statistics = None
        if dtype == 'numeric' and len(non_null) > 0:
            statistics = {
                'mean': float(non_null.mean()),
                'std': float(non_null.std() if len(non_null) > 1 else 0),
                'min': float(non_null.min()),
                'max': float(non_null.max()),
                'q25': float(non_null.quantile(0.25)),
                'q75': float(non_null.quantile(0.75)),
            }
        
        return DataColumnMeta(
            name=col_name,
            dtype=dtype,
            nullable=series.isna().any(),
            missing_count=int(series.isna().sum()),
            unique_count=int(non_null.nunique()),
            sample_values=non_null.astype(str).head(5).tolist(),
            statistics=statistics
        )
    
    def _is_smiles_series(self, series: pd.Series, threshold: float = 0.8) -> bool:
        """Check if series likely contains SMILES strings"""
        if len(series) == 0: return False
        sample_size = min(100, len(series))
        sample = series.sample(sample_size, random_state=42)
        valid_count = sum(1 for s in sample if validate_smiles_batch([str(s)])[0])
        return valid_count / sample_size >= threshold
    
    def _is_smiles_candidate(self, series: pd.Series) -> bool:
        """Heuristic check for SMILES column (name pattern + content)"""
        col_name_lower = series.name.lower() if series.name else ''
        name_indicators = ['smiles', 'smi', 'structure', 'mol', 'compound']
        if any(ind in col_name_lower for ind in name_indicators):
            return True
        return self._is_smiles_series(series, threshold=0.5)
    
    def _check_duplicate(self, file_hash: str) -> bool:
        """Check if dataset with same hash already exists"""
        return file_hash in self._metadata_cache
    
    async def _read_file(self, file: UploadFile) -> pd.DataFrame:
        """Read uploaded file with format auto-detection"""
        filename = file.filename.lower()
        
        if filename.endswith('.csv'):
            return pd.read_csv(file.file)
        elif filename.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file.file)
        elif filename.endswith('.parquet'):
            return pd.read_parquet(file.file)
        elif filename.endswith('.sdf'):
            from rdkit.Chem import PandasTools
            return PandasTools.LoadSDF(file.file)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported file format: {filename}")
    
    def get_dataset(self, dataset_id: str) -> pd.DataFrame:
        """Load dataset by ID"""
        file_path = self.storage_path / f"{dataset_id}.parquet"
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="Dataset not found")
        return pd.read_parquet(file_path)
    
    def get_metadata(self, dataset_id: str) -> DatasetMeta:
        """Get dataset metadata"""
        if dataset_id in self._metadata_cache:
            return self._metadata_cache[dataset_id]
        
        meta_path = self.storage_path / f"{dataset_id}_meta.json"
        if not meta_path.exists():
            raise HTTPException(status_code=404, detail="Metadata not found")
        
        with open(meta_path, 'r', encoding='utf-8') as f:
            meta_dict = json.load(f)
        return DatasetMeta(**meta_dict)
