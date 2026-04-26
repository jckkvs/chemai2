# backend/data/data_loader.py — 精緻化版 (データ読み込みエンジン)

from typing import Union, Optional, Dict, List, Tuple, BinaryIO
from pathlib import Path
import logging
import warnings
import io

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_data(
    source: Union[str, Path, BinaryIO],
    file_format: Optional[str] = None,
    encoding: Optional[str] = None,
    chunk_size: Optional[int] = None,
    low_memory: bool = True,
    dtype_mapping: Optional[Dict[str, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from various formats with robust encoding detection and memory management
    """
    # 【修正点3】形式の自動検出: 拡張子よりコンテンツを優先
    detected_format = file_format or _detect_format_from_content(source)
    
    if detected_format in ('csv', 'tsv', 'txt'):
        return _load_text_file(
            source, 
            delimiter=',' if detected_format == 'csv' else '\t',
            encoding=encoding,
            chunk_size=chunk_size,
            low_memory=low_memory,
            dtype_mapping=dtype_mapping,
            **kwargs
        )
    
    elif detected_format in ('excel', 'xlsx', 'xls'):
        return _load_excel_file(source, **kwargs)
    
    elif detected_format == 'parquet':
        return _load_parquet_file(source, **kwargs)
    
    elif detected_format == 'json':
        return _load_json_file(source, encoding=encoding, **kwargs)
    
    elif detected_format == 'sdf':
        return _load_sdf_file(source, **kwargs)
    
    else:
        raise ValueError(f"Unsupported or unrecognized format: {detected_format}")


def _detect_format_from_content(source: Union[str, Path, BinaryIO]) -> str:
    """
    Detect file format from content rather than just extension
    """
    if isinstance(source, (str, Path)):
        path = Path(source)
        ext = path.suffix.lower().lstrip('.')
        ext_map = {
            'csv': 'csv', 'tsv': 'tsv', 'txt': 'csv',
            'xlsx': 'excel', 'xls': 'excel', 'xlsm': 'excel',
            'parquet': 'parquet', 'pq': 'parquet',
            'json': 'json', 'jsonl': 'json',
            'sdf': 'sdf', 'mol': 'sdf',
        }
        if ext in ext_map:
            try:
                with open(path, 'rb') as f:
                    header = f.read(256)
                return _detect_from_bytes(header, ext)
            except Exception:
                return ext_map[ext]
        return 'csv'
    
    elif hasattr(source, 'read'):
        pos = source.tell() if hasattr(source, 'tell') else None
        try:
            header = source.read(256)
            if pos is not None and hasattr(source, 'seek'):
                source.seek(pos)
            return _detect_from_bytes(header)
        except Exception:
            if pos is not None and hasattr(source, 'seek'):
                source.seek(pos)
            return 'csv'
    
    return 'csv'


def _detect_from_bytes(header: bytes, hint: Optional[str] = None) -> str:
    """Detect format from file header bytes"""
    if header[:4] == b'PAR1' or header[-4:] == b'PAR1':
        return 'parquet'
    if header[:8] == b'\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1':
        return 'excel'
    if header[:2] == b'PK':
        if b'xl/' in header[:256] or b'[Content_Types].xml' in header[:256]:
            return 'excel'
        if hint == 'parquet' or b'parquet' in header[:256].lower():
            return 'parquet'
        return 'excel'
    if header[:4] == b'$$$$' or (b'\n' in header[:100] and b'  ' in header[:20]):
        return 'sdf'
    stripped = header.lstrip()
    if stripped[:1] in (b'{', b'['):
        return 'json'
    try:
        sample = header.decode('utf-8', errors='ignore')
        lines = sample.split('\n')[:5]
        comma_count = sum(line.count(',') for line in lines)
        tab_count = sum(line.count('\t') for line in lines)
        if tab_count > comma_count and tab_count >= len(lines):
            return 'tsv'
        elif comma_count >= len(lines):
            return 'csv'
    except Exception:
        pass
    return hint or 'csv'


def _load_text_file(
    source: Union[str, Path, BinaryIO],
    delimiter: str = ',',
    encoding: Optional[str] = None,
    chunk_size: Optional[int] = None,
    low_memory: bool = True,
    dtype_mapping: Optional[Dict[str, str]] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load CSV/TSV/TXT files with robust encoding detection
    """
    encodings_to_try = []
    if encoding:
        encodings_to_try.append(encoding)
    encodings_to_try.extend(['utf-8', 'utf-8-sig', 'shift_jis', 'euc-jp', 'cp932', 'latin1'])
    
    last_error = None
    for enc in encodings_to_try:
        try:
            if chunk_size and chunk_size > 0:
                chunks = []
                for chunk in pd.read_csv(
                    source, delimiter=delimiter, encoding=enc,
                    chunksize=chunk_size, low_memory=low_memory,
                    dtype=dtype_mapping, **kwargs
                ):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            else:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore', category=pd.errors.DtypeWarning)
                    df = pd.read_csv(
                        source, delimiter=delimiter, encoding=enc,
                        low_memory=low_memory, dtype=dtype_mapping, **kwargs
                    )
            return df
        except UnicodeDecodeError as e:
            last_error = e
            continue
        except Exception:
            raise
    raise ValueError(f"Failed to load file with any encoding. Last error: {last_error}")


def _load_excel_file(source: Union[str, Path, BinaryIO], **kwargs) -> pd.DataFrame:
    """Load Excel files with sheet handling and memory optimization"""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        df = pd.read_excel(source, **kwargs)
    return df


def _load_parquet_file(source: Union[str, Path, BinaryIO], **kwargs) -> pd.DataFrame:
    """Load Parquet files with engine fallback"""
    engines_to_try = kwargs.pop('engine', ['pyarrow', 'fastparquet'])
    if isinstance(engines_to_try, str): engines_to_try = [engines_to_try]
    last_error = None
    for engine in engines_to_try:
        try: return pd.read_parquet(source, engine=engine, **kwargs)
        except (ImportError, Exception) as e: last_error = e; continue
    raise ValueError(f"Failed to load Parquet with any engine. Last error: {last_error}")


def _load_json_file(source: Union[str, Path, BinaryIO], encoding: Optional[str] = None, **kwargs) -> pd.DataFrame:
    """Load JSON/JSONL files with encoding handling"""
    encodings = [encoding] if encoding else ['utf-8', 'utf-8-sig', 'latin1']
    for enc in encodings:
        try: return pd.read_json(source, encoding=enc, **kwargs)
        except (UnicodeDecodeError, ValueError): continue
    raise ValueError("Failed to load JSON with any encoding")


def _load_sdf_file(source: Union[str, Path, BinaryIO], **kwargs) -> pd.DataFrame:
    """Load SDF/MOL files via RDKit with error handling"""
    try:
        from rdkit import Chem
        from rdkit.Chem import PandasTools
    except ImportError:
        raise ImportError("RDKit is required for SDF file support")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            df = PandasTools.LoadSDF(str(source) if isinstance(source, (str, Path)) else source, **kwargs)
        return df if df is not None else pd.DataFrame()
    except Exception as e:
        logger.error(f"Failed to load SDF file: {e}")
        raise
