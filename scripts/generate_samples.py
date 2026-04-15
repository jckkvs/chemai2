#!/usr/bin/env python3
"""
サンプルデータ生成スクリプト
- SMILES 系データ：RDKit 記述子から擬似目的変数を生成
- テーブルデータ：scikit-learn の make_regression + 人工的ノイズ・欠損値追加
- 混合物データ：化合物 3 列（SMILES）、回帰、重量 WT％での分率
"""
import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from sklearn.datasets import make_regression, make_classification
import os

# ========== 固定デバッグデータ (ユーザー提供) ==========
MIXTURE_SMILES_ONLY_DATA = """Compound_1_Name,Compound_1_SMILES,Compound_1_WT%,Compound_2_Name,Compound_2_SMILES,Compound_2_WT%,Compound_3_Name,Compound_3_SMILES,Compound_3_WT%,Compound_4_Name,Compound_4_SMILES,Compound_4_WT%,Sample_ID,Target_BoilingPoint_C
Cyclohexane,C1CCCCC1,15.14,Isopropanol,CC(C)C,11.49,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,41.63,Ethanol,CCO,31.74,MIX_SMILES_0001,64.56
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,54.38,Benzene,c1ccccc1,45.62,,,,,,,MIX_SMILES_0002,84.65
Ethanol,CCO,34.85,Isopropanol,CC(C)C,12.27,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,35.48,Aspirin,CC(=O)Oc1ccccc1C(=O)O,17.4,MIX_SMILES_0003,71.5
Isopropanol,CC(C)C,36.89,Ethanol,CCO,16.19,Benzene,c1ccccc1,29.04,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,17.88,MIX_SMILES_0004,46.59
Ethanol,CCO,27.75,Cyclohexane,C1CCCCC1,25.59,Acetic Acid,CC(=O)O,46.66,,,,MIX_SMILES_0005,28.01
Benzene,c1ccccc1,27.79,Isopropanol,CC(C)C,29.45,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,42.77,,,,MIX_SMILES_0006,76.98
Isopropanol,CC(C)C,69.5,Aspirin,CC(=O)Oc1ccccc1C(=O)O,30.5,,,,,,,MIX_SMILES_0007,55.97
Acetic Acid,CC(=O)O,60.05,Aspirin,CC(=O)Oc1ccccc1C(=O)O,39.95,,,,,,,MIX_SMILES_0008,65.76
Cyclohexane,C1CCCCC1,21.55,Ethanol,CCO,16.81,Isopropanol,CC(C)C,32.45,Aspirin,CC(=O)Oc1ccccc1C(=O)O,29.19,MIX_SMILES_0009,48.72
Aspirin,CC(=O)Oc1ccccc1C(=O)O,16.25,Cyclohexane,C1CCCCC1,51.17,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,32.58,,,,MIX_SMILES_0010,80.61
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,29.03,Isopropanol,CC(C)C,32.31,Aspirin,CC(=O)Oc1ccccc1C(=O)O,38.66,,,,MIX_SMILES_0011,82.17
Ethanol,CCO,42.74,Benzene,c1ccccc1,57.26,,,,,,,MIX_SMILES_0012,31.99
Aspirin,CC(=O)Oc1ccccc1C(=O)O,29.37,Isopropanol,CC(C)C,70.63,,,,,,,MIX_SMILES_0013,49.42
Acetic Acid,CC(=O)O,22.57,Aspirin,CC(=O)Oc1ccccc1C(=O)O,19.11,Isopropanol,CC(C)C,21.41,Benzene,c1ccccc1,36.91,MIX_SMILES_0014,47.63
Benzene,c1ccccc1,57.56,Acetic Acid,CC(=O)O,42.44,,,,,,,MIX_SMILES_0015,38.06
Cyclohexane,C1CCCCC1,34.23,Aspirin,CC(=O)Oc1ccccc1C(=O)O,30.58,Isopropanol,CC(C)C,35.19,,,,MIX_SMILES_0016,63.75
Cyclohexane,C1CCCCC1,19.49,Isopropanol,CC(C)C,27.84,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,27.88,Ethanol,CCO,24.79,MIX_SMILES_0017,53.14
Ethanol,CCO,61.53,Isopropanol,CC(C)C,38.47,,,,,,,MIX_SMILES_0018,19.07
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,18.15,Benzene,c1ccccc1,11.16,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,36.07,Cyclohexane,C1CCCCC1,34.62,MIX_SMILES_0019,89.44
Benzene,c1ccccc1,12.99,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,43.44,Isopropanol,CC(C)C,43.56,,,,MIX_SMILES_0020,71.04
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,33.51,Isopropanol,CC(C)C,30.35,Aspirin,CC(=O)Oc1ccccc1C(=O)O,26.29,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,9.85,MIX_SMILES_0021,92.28
Ethanol,CCO,74.5,Aspirin,CC(=O)Oc1ccccc1C(=O)O,25.5,,,,,,,MIX_SMILES_0022,35.81
Acetic Acid,CC(=O)O,35.75,Isopropanol,CC(C)C,64.25,,,,,,,MIX_SMILES_0023,29.12
Aspirin,CC(=O)Oc1ccccc1C(=O)O,32.41,Ethanol,CCO,35.61,Isopropanol,CC(C)C,31.98,,,,MIX_SMILES_0024,46.52
Acetic Acid,CC(=O)O,26.4,Isopropanol,CC(C)C,73.6,,,,,,,MIX_SMILES_0025,30.95
Isopropanol,CC(C)C,36.58,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,63.42,,,,,,,MIX_SMILES_0026,95.64
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,40.97,Aspirin,CC(=O)Oc1ccccc1C(=O)O,11.12,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,15.93,Benzene,c1ccccc1,31.98,MIX_SMILES_0027,92.47
Aspirin,CC(=O)Oc1ccccc1C(=O)O,27.55,Acetic Acid,CC(=O)O,24.57,Benzene,c1ccccc1,47.88,,,,MIX_SMILES_0028,54.43
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,38.34,Acetic Acid,CC(=O)O,19.4,Benzene,c1ccccc1,14.02,Cyclohexane,C1CCCCC1,28.25,MIX_SMILES_0029,71.83
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,27.58,Aspirin,CC(=O)Oc1ccccc1C(=O)O,43.23,Cyclohexane,C1CCCCC1,14.42,Isopropanol,CC(C)C,14.77,MIX_SMILES_0030,90.22
Cyclohexane,C1CCCCC1,43.24,Isopropanol,CC(C)C,19.75,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,37.01,,,,MIX_SMILES_0031,78.5
Ethanol,CCO,11.12,Benzene,c1ccccc1,24.79,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,26.11,Acetic Acid,CC(=O)O,37.98,MIX_SMILES_0032,63.04
Acetic Acid,CC(=O)O,31.83,Benzene,c1ccccc1,30.56,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,37.61,,,,MIX_SMILES_0033,75.67
Aspirin,CC(=O)Oc1ccccc1C(=O)O,26.73,Benzene,c1ccccc1,73.27,,,,,,,MIX_SMILES_0034,58.59
Aspirin,CC(=O)Oc1ccccc1C(=O)O,9.46,Benzene,c1ccccc1,35.96,Acetic Acid,CC(=O)O,31.02,Isopropanol,CC(C)C,23.56,MIX_SMILES_0035,42.18
Acetic Acid,CC(=O)O,38.53,Cyclohexane,C1CCCCC1,36.53,Isopropanol,CC(C)C,16.42,Benzene,c1ccccc1,8.53,MIX_SMILES_0036,33.64
Acetic Acid,CC(=O)O,8.84,Cyclohexane,C1CCCCC1,26.72,Benzene,c1ccccc1,38.05,Aspirin,CC(=O)Oc1ccccc1C(=O)O,26.39,MIX_SMILES_0037,61.47
Isopropanol,CC(C)C,51.18,Ethanol,CCO,48.82,,,,,,,MIX_SMILES_0038,18.96
Acetic Acid,CC(=O)O,74.39,Cyclohexane,C1CCCCC1,25.61,,,,,,,MIX_SMILES_0039,38.94
Aspirin,CC(=O)Oc1ccccc1C(=O)O,28.85,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,57.21,Acetic Acid,CC(=O)O,13.94,,,,MIX_SMILES_0040,108.15
Acetic Acid,CC(=O)O,49.77,Benzene,c1ccccc1,50.23,,,,,,,MIX_SMILES_0041,35.67
Isopropanol,CC(C)C,49.91,Benzene,c1ccccc1,50.09,,,,,,,MIX_SMILES_0042,36.16
Benzene,c1ccccc1,61.01,Isopropanol,CC(C)C,38.99,,,,,,,MIX_SMILES_0043,34.87
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,26.57,Cyclohexane,C1CCCCC1,28.3,Ethanol,CCO,45.12,,,,MIX_SMILES_0044,55.58
Benzene,c1ccccc1,80.29,Cyclohexane,C1CCCCC1,19.71,,,,,,,MIX_SMILES_0045,40.97
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,33.3,Benzene,c1ccccc1,32.45,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,20.16,Cyclohexane,C1CCCCC1,14.09,MIX_SMILES_0046,92.13
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,36.76,Acetic Acid,CC(=O)O,63.24,,,,,,,MIX_SMILES_0047,70.79
Isopropanol,CC(C)C,39.67,Ethanol,CCO,60.33,,,,,,,MIX_SMILES_0048,20.39
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,51.56,Cyclohexane,C1CCCCC1,21.98,Aspirin,CC(=O)Oc1ccccc1C(=O)O,26.46,,,,MIX_SMILES_0049,104.57
Isopropanol,CC(C)C,70.26,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,29.74,,,,,,,MIX_SMILES_0050,55.77"""

MIXTURE_SMILES_NUMERIC_DATA = """Compound_1_Name,Compound_1_SMILES,Compound_1_WT%,Compound_2_Name,Compound_2_SMILES,Compound_2_WT%,Compound_3_Name,Compound_3_SMILES,Compound_3_WT%,Compound_4_Name,Compound_4_SMILES,Compound_4_WT%,Pressure_atm,ReactionTime_h,Sample_ID,StirringSpeed_rpm,Target_Yield_pct,Temperature_C,pH
Aspirin,CC(=O)Oc1ccccc1C(=O)O,54.77,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,45.23,,,,,,,0.63,19.35,MIX_SMILES_NUM_0001,455,91.77,63.3,7.87
Isopropanol,CC(C)C,32.88,Aspirin,CC(=O)Oc1ccccc1C(=O)O,24.69,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,42.44,,,,2.43,5.13,MIX_SMILES_NUM_0002,307,81.66,64.4,3.12
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,19.99,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,27.7,Acetic Acid,CC(=O)O,26.13,Benzene,c1ccccc1,26.17,0.97,10.18,MIX_SMILES_NUM_0003,957,83.71,62.0,3.45
Benzene,c1ccccc1,34.48,Isopropanol,CC(C)C,65.52,,,,,,,2.5,17.58,MIX_SMILES_NUM_0004,470,76.12,78.9,6.49
Ethanol,CCO,50.38,Benzene,c1ccccc1,43.2,Cyclohexane,C1CCCCC1,6.42,,,,3.15,10.74,MIX_SMILES_NUM_0005,920,95.16,33.5,8.83
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,38.65,Ethanol,CCO,11.36,Aspirin,CC(=O)Oc1ccccc1C(=O)O,49.99,,,,1.33,43.62,MIX_SMILES_NUM_0006,211,96.38,30.3,5.08
Acetic Acid,CC(=O)O,22.02,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,27.5,Benzene,c1ccccc1,26.74,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,23.74,2.39,26.14,MIX_SMILES_NUM_0007,868,81.33,98.6,9.25
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,27.81,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,28.67,Ethanol,CCO,43.52,,,,1.26,38.93,MIX_SMILES_NUM_0008,574,90.47,40.1,5.14
Benzene,c1ccccc1,52.22,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,47.78,,,,,,,2.32,12.22,MIX_SMILES_NUM_0009,472,99.8,20.3,4.96
Ethanol,CCO,42.74,Isopropanol,CC(C)C,57.26,,,,,,,4.06,19.26,MIX_SMILES_NUM_0010,483,86.2,33.3,4.3
Isopropanol,CC(C)C,19.38,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,52.54,Ethanol,CCO,28.08,,,,2.46,12.87,MIX_SMILES_NUM_0011,595,74.92,54.9,2.44
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,22.92,Acetic Acid,CC(=O)O,18.84,Aspirin,CC(=O)Oc1ccccc1C(=O)O,47.34,Ethanol,CCO,10.9,0.73,42.14,MIX_SMILES_NUM_0012,809,92.35,39.0,8.81
Isopropanol,CC(C)C,46.54,Aspirin,CC(=O)Oc1ccccc1C(=O)O,21.96,Cyclohexane,C1CCCCC1,31.5,,,,0.62,40.94,MIX_SMILES_NUM_0013,833,96.53,59.3,2.48
Aspirin,CC(=O)Oc1ccccc1C(=O)O,15.65,Acetic Acid,CC(=O)O,26.79,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,20.89,Cyclohexane,C1CCCCC1,36.67,1.0,46.12,MIX_SMILES_NUM_0014,942,88.72,50.7,5.55
Aspirin,CC(=O)Oc1ccccc1C(=O)O,18.87,Ethanol,CCO,46.53,Isopropanol,CC(C)C,34.6,,,,3.49,43.23,MIX_SMILES_NUM_0015,108,79.08,94.9,11.33
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,42.15,Isopropanol,CC(C)C,57.85,,,,,,,4.75,41.97,MIX_SMILES_NUM_0016,613,99.23,78.2,11.69
Cyclohexane,C1CCCCC1,21.93,Ethanol,CCO,17.47,Benzene,c1ccccc1,32.78,Aspirin,CC(=O)Oc1ccccc1C(=O)O,27.82,0.86,9.58,MIX_SMILES_NUM_0017,456,86.9,82.9,3.88
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,59.08,Isopropanol,CC(C)C,17.84,Benzene,c1ccccc1,23.08,,,,4.07,32.18,MIX_SMILES_NUM_0018,487,70.93,89.5,3.67
Cyclohexane,C1CCCCC1,54.51,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,29.98,Ethanol,CCO,15.51,,,,1.17,16.89,MIX_SMILES_NUM_0019,252,70.65,26.5,5.13
Cyclohexane,C1CCCCC1,34.18,Benzene,c1ccccc1,10.0,Aspirin,CC(=O)Oc1ccccc1C(=O)O,23.32,Ethanol,CCO,32.51,3.29,6.59,MIX_SMILES_NUM_0020,772,99.34,31.7,4.09
Ethanol,CCO,52.42,Isopropanol,CC(C)C,15.71,Aspirin,CC(=O)Oc1ccccc1C(=O)O,31.86,,,,0.95,3.13,MIX_SMILES_NUM_0021,833,83.84,21.7,2.15
Cyclohexane,C1CCCCC1,47.88,Isopropanol,CC(C)C,28.61,Aspirin,CC(=O)Oc1ccccc1C(=O)O,23.51,,,,2.5,14.63,MIX_SMILES_NUM_0022,954,82.88,88.4,11.45
Isopropanol,CC(C)C,15.35,Aspirin,CC(=O)Oc1ccccc1C(=O)O,56.7,Cyclohexane,C1CCCCC1,27.95,,,,2.35,27.58,MIX_SMILES_NUM_0023,710,74.91,62.1,8.96
Aspirin,CC(=O)Oc1ccccc1C(=O)O,31.95,Ethanol,CCO,68.05,,,,,,,1.38,15.1,MIX_SMILES_NUM_0024,428,88.49,83.0,3.69
Ethanol,CCO,25.86,Cyclohexane,C1CCCCC1,28.71,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,45.43,,,,3.15,19.98,MIX_SMILES_NUM_0025,835,76.65,71.2,11.96
Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,81.47,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,18.53,,,,,,,4.48,5.43,MIX_SMILES_NUM_0026,108,79.52,78.2,9.39
Cyclohexane,C1CCCCC1,29.83,Benzene,c1ccccc1,34.86,Aspirin,CC(=O)Oc1ccccc1C(=O)O,35.31,,,,0.5,3.95,MIX_SMILES_NUM_0027,511,79.05,42.4,5.2
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,35.84,Cyclohexane,C1CCCCC1,16.5,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,47.66,,,,3.69,32.32,MIX_SMILES_NUM_0028,829,86.07,72.7,3.19
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,14.73,Aspirin,CC(=O)Oc1ccccc1C(=O)O,29.98,Isopropanol,CC(C)C,11.08,Cyclohexane,C1CCCCC1,44.22,3.31,23.51,MIX_SMILES_NUM_0029,918,89.5,82.4,11.43
Aspirin,CC(=O)Oc1ccccc1C(=O)O,25.33,Acetic Acid,CC(=O)O,32.48,Cyclohexane,C1CCCCC1,16.51,Isopropanol,CC(C)C,25.68,1.3,31.7,MIX_SMILES_NUM_0030,736,83.27,99.9,10.63
Acetic Acid,CC(=O)O,18.4,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,32.55,Cyclohexane,C1CCCCC1,24.4,Ethanol,CCO,24.65,1.79,33.19,MIX_SMILES_NUM_0031,489,92.51,80.8,9.45
Ethanol,CCO,18.33,Benzene,c1ccccc1,17.47,Aspirin,CC(=O)Oc1ccccc1C(=O)O,43.23,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,20.97,4.03,26.54,MIX_SMILES_NUM_0032,606,94.02,23.3,4.9
Isopropanol,CC(C)C,86.86,Ethanol,CCO,13.14,,,,,,,0.78,14.74,MIX_SMILES_NUM_0033,391,73.07,77.5,7.74
Isopropanol,CC(C)C,63.15,Aspirin,CC(=O)Oc1ccccc1C(=O)O,36.85,,,,,,,4.18,17.03,MIX_SMILES_NUM_0034,204,96.38,62.5,4.39
Aspirin,CC(=O)Oc1ccccc1C(=O)O,46.12,Cyclohexane,C1CCCCC1,14.15,Isopropanol,CC(C)C,13.25,Acetic Acid,CC(=O)O,26.48,3.22,32.33,MIX_SMILES_NUM_0035,595,74.96,93.2,5.71
Cyclohexane,C1CCCCC1,20.08,Acetic Acid,CC(=O)O,21.96,Benzene,c1ccccc1,57.96,,,,3.66,41.97,MIX_SMILES_NUM_0036,547,75.29,67.6,4.52
Cyclohexane,C1CCCCC1,54.89,Ethanol,CCO,24.97,Acetic Acid,CC(=O)O,20.14,,,,1.19,25.7,MIX_SMILES_NUM_0037,971,97.74,93.6,9.66
Ethanol,CCO,51.81,Isopropanol,CC(C)C,48.19,,,,,,,0.93,16.59,MIX_SMILES_NUM_0038,646,95.53,88.9,8.74
Cyclohexane,C1CCCCC1,18.06,Aspirin,CC(=O)Oc1ccccc1C(=O)O,49.2,Acetic Acid,CC(=O)O,32.74,,,,2.39,37.31,MIX_SMILES_NUM_0039,890,92.51,69.5,8.19
Acetic Acid,CC(=O)O,18.17,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,70.62,Aspirin,CC(=O)Oc1ccccc1C(=O)O,11.21,,,,1.68,13.92,MIX_SMILES_NUM_0040,147,75.0,79.9,11.02
Acetic Acid,CC(=O)O,56.57,Benzene,c1ccccc1,43.43,,,,,,,3.93,10.6,MIX_SMILES_NUM_0041,313,99.99,35.2,5.2
Isopropanol,CC(C)C,57.25,Benzene,c1ccccc1,42.75,,,,,,,3.0,17.2,MIX_SMILES_NUM_0042,279,71.05,42.4,5.82
Benzene,c1ccccc1,19.33,Isopropanol,CC(C)C,80.67,,,,,,,2.56,9.54,MIX_SMILES_NUM_0043,993,73.5,91.8,7.91
Cyclohexane,C1CCCCC1,31.42,Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,11.5,Ethanol,CCO,57.08,,,,4.15,19.36,MIX_SMILES_NUM_0044,307,81.42,88.4,11.39
Benzene,c1ccccc1,69.54,Cyclohexane,C1CCCCC1,30.46,,,,,,,2.31,16.8,MIX_SMILES_NUM_0045,433,96.38,36.5,4.3
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,31.25,Cyclohexane,C1CCCCC1,54.21,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,14.54,,,,2.5,14.63,MIX_SMILES_NUM_0046,954,82.88,88.4,11.45
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,15.35,Aspirin,CC(=O)Oc1ccccc1C(=O)O,56.7,Cyclohexane,C1CCCCC1,27.95,,,,2.35,27.58,MIX_SMILES_NUM_0047,710,74.91,62.1,8.96
Isopropanol,CC(C)C,31.95,Ethanol,CCO,68.05,,,,,,,1.38,15.1,MIX_SMILES_NUM_0048,428,88.49,83.0,3.69
Ibuprofen,CC(C)Cc1ccc(cc1)C(C)C(=O)O,25.86,Cyclohexane,C1CCCCC1,28.71,Aspirin,CC(=O)Oc1ccccc1C(=O)O,45.43,,,,3.15,19.98,MIX_SMILES_NUM_0049,835,76.65,71.2,11.96
Isopropanol,CC(C)C,81.47,Caffeine,CN1=C(C(N(C1=O)=O)C)N(C)C,18.53,,,,,,,4.48,5.43,MIX_SMILES_NUM_0050,108,79.52,78.2,9.39"""



# ========== SMILES データ生成 ==========
SMILES_POOL = [
    # 医薬品
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(C)CC1=CC=C(C=C1)C(C)C(=O)O",  # Naproxen
    # 溶媒・試薬
    "CCO", "CC(C)O", "CC(C)(C)O", "c1ccccc1", "CC1=CC=CC=C1",
    # 天然物・複雑構造
    "CC1=CC2=C(C=C1C=O)C(=O)C3=C(O2)C=CC(=C3)O",  # Simple flavonoid
    "CCC1=CC(=C(C=C1)O)C=O",  # Vanillin
    # 極端ケース
    "C"*50,  # 長鎖アルカン（エラーテスト）
    "C1CC1C1CC1C1CC1",  # 複雑な環構造
]

def calc_pseudo_targets(smiles: str) -> dict:
    """SMILES から擬似目的変数を計算（物性値の相関を模擬）"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"logS": np.nan, "pIC50": np.nan, "class": -1}

    # RDKit 記述子
    mw = Descriptors.MolWt(mol)
    logp = Crippen.MolLogP(mol)
    tpsa = Descriptors.TPSA(mol)
    hbd = Lipinski.NumHDonors(mol)

    # 擬似 logS（水溶性）: MW↑・logP↑・TPSA↑ → 溶解度↓
    logS = -0.01*mw - 0.3*logp - 0.005*tpsa + np.random.normal(0, 0.3)

    # 擬似 pIC50（活性）: 複雑な構造ほど高い活性を模擬
    pIC50 = 3 + 0.005*mw + 0.2*logp - 0.1*hbd + np.random.normal(0, 0.5)

    # 分類ラベル（logP 閾値ベース）
    cls = 1 if logp > 2.5 else 0

    return {
        "logS": round(logS, 3),
        "pIC50": round(pIC50, 3),
        "class": cls,
        "MW": round(mw, 2),
        "LogP": round(logp, 2)
    }

def generate_smiles_samples(n: int, filename: str):
    data = []
    for i in range(n):
        smiles = SMILES_POOL[i % len(SMILES_POOL)]
        targets = calc_pseudo_targets(smiles)
        data.append({
            "SMILES": smiles,
            "Compound_Name": f"Compound_{i+1:03d}",
            "logS": targets["logS"],
            "pIC50": targets["pIC50"],
            "Class": targets["class"],
            "Source": "Synthetic",
            "Notes": "Generated for testing"
        })
    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 混合物データ生成 ==========
def generate_mixture_samples(n: int, filename: str):
    """
    混合物データ生成：化合物 3 列（SMILES）、回帰目的変数、重量％分率（WT%）
    - 3 成分の混合物を想定
    - 各成分の重量％は合計 100% になるように正規化
    - 目的変数は各成分の寄与と相互作用から擬似生成
    """
    # 使用する化合物プール（異なる特性を持つものを選択）
    compound_pool = [
        ("CCO", "Ethanol"),                    # 極性溶媒
        ("CC(C)O", "Isopropanol"),             # 極性溶媒
        ("c1ccccc1", "Benzene"),               # 非極性溶媒
        ("CC1=CC=CC=C1", "Toluene"),           # 非極性溶媒
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),  # 医薬品
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),  # 天然物
        ("CCC1=CC(=C(C=C1)O)C=O", "Vanillin"), # 香料
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),  # 医薬品
    ]

    data = []
    for i in range(n):
        # 3 成分をランダム選択（重複あり）
        indices = np.random.choice(len(compound_pool), size=3, replace=True)

        # 重量％の生成（合計 100% に正規化）
        raw_weights = np.random.uniform(10, 90, size=3)
        wt_percent = (raw_weights / raw_weights.sum() * 100).round(2)

        # 化合物情報
        smiles_list = [compound_pool[idx][0] for idx in indices]
        names = [compound_pool[idx][1] for idx in indices]

        # 擬似目的変数の生成（各成分の特性と相互作用を考慮）
        mol_props = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                mol_props.append((mw, logp, tpsa))
            else:
                mol_props.append((0, 0, 0))

        # 重量％を係数として使用（0-1 の範囲）
        frac = wt_percent / 100.0

        # 擬似物性値：各成分の寄与の加重平均 + 相互作用項 + ノイズ
        base_property = sum(frac[j] * (mol_props[j][1] * 0.5 + mol_props[j][2] * 0.01) for j in range(3))
        interaction = frac[0] * frac[1] * 5 + frac[1] * frac[2] * 3  # 二元相互作用
        target = base_property + interaction + np.random.normal(0, 2)

        data.append({
            "Compound_1_SMILES": smiles_list[0],
            "Compound_1_Name": names[0],
            "Compound_1_WT%": wt_percent[0],
            "Compound_2_SMILES": smiles_list[1],
            "Compound_2_Name": names[1],
            "Compound_2_WT%": wt_percent[1],
            "Compound_3_SMILES": smiles_list[2],
            "Compound_3_Name": names[2],
            "Compound_3_WT%": wt_percent[2],
            "Target_Property": round(target, 3),
            "Sample_ID": f"MIX{i+1:04d}",
            "Total_WT%": wt_percent.sum(),  # 検証用（常に 100%）
            "Notes": "Synthetic mixture data"
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")


def generate_mixture_samples_with_numeric(n: int, filename: str):
    """
    混合物データ生成（デバッグ用）：化合物 3 列（SMILES）、回帰目的変数、重量％分率（WT%）
    ＋数値データ（温度、湿度など）
    - 3 成分の混合物を想定
    - 各成分の重量％は合計 100% になるように正規化
    - 目的変数は各成分の寄与と相互作用＋数値特徴量から擬似生成
    - 温度、湿度、圧力、pH などの数値特徴量を追加
    """
    # 使用する化合物プール（異なる特性を持つものを選択）
    compound_pool = [
        ("CCO", "Ethanol"),                    # 極性溶媒
        ("CC(C)O", "Isopropanol"),             # 極性溶媒
        ("c1ccccc1", "Benzene"),               # 非極性溶媒
        ("CC1=CC=CC=C1", "Toluene"),           # 非極性溶媒
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),  # 医薬品
        ("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "Caffeine"),  # 天然物
        ("CCC1=CC(=C(C=C1)O)C=O", "Vanillin"), # 香料
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),  # 医薬品
    ]

    data = []
    for i in range(n):
        # 3 成分をランダム選択（重複あり）
        indices = np.random.choice(len(compound_pool), size=3, replace=True)

        # 重量％の生成（合計 100% に正規化）
        raw_weights = np.random.uniform(10, 90, size=3)
        wt_percent = (raw_weights / raw_weights.sum() * 100).round(2)

        # 化合物情報
        smiles_list = [compound_pool[idx][0] for idx in indices]
        names = [compound_pool[idx][1] for idx in indices]

        # 数値特徴量の生成（温度、湿度、圧力、pH など）
        temperature = np.random.uniform(20, 80)  # ℃
        humidity = np.random.uniform(30, 90)     # %
        pressure = np.random.uniform(0.8, 2.0)   # atm
        ph_value = np.random.uniform(4, 10)      # pH
        stirring_speed = np.random.uniform(100, 1000)  # rpm
        reaction_time = np.random.uniform(0.5, 24)     # hours

        # 擬似目的変数の生成（各成分の特性と相互作用＋数値特徴量を考慮）
        mol_props = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                mw = Descriptors.MolWt(mol)
                logp = Crippen.MolLogP(mol)
                tpsa = Descriptors.TPSA(mol)
                mol_props.append((mw, logp, tpsa))
            else:
                mol_props.append((0, 0, 0))

        # 重量％を係数として使用（0-1 の範囲）
        frac = wt_percent / 100.0

        # 擬似物性値：各成分の寄与の加重平均 + 相互作用項 + 数値特徴量の影響 + ノイズ
        base_property = sum(frac[j] * (mol_props[j][1] * 0.5 + mol_props[j][2] * 0.01) for j in range(3))
        interaction = frac[0] * frac[1] * 5 + frac[1] * frac[2] * 3  # 二元相互作用

        # 数値特徴量の効果（温度上昇で増加、pH で最適値など）
        temp_effect = (temperature - 50) * 0.05  # 50℃を基準
        ph_effect = -0.1 * (ph_value - 7) ** 2   # pH7 で最大
        pressure_effect = pressure * 0.3
        time_effect = np.log(reaction_time + 1) * 0.5

        target = base_property + interaction + temp_effect + ph_effect + pressure_effect + time_effect + np.random.normal(0, 1.5)

        data.append({
            "Compound_1_SMILES": smiles_list[0],
            "Compound_1_Name": names[0],
            "Compound_1_WT%": wt_percent[0],
            "Compound_2_SMILES": smiles_list[1],
            "Compound_2_Name": names[1],
            "Compound_2_WT%": wt_percent[1],
            "Compound_3_SMILES": smiles_list[2],
            "Compound_3_Name": names[2],
            "Compound_3_WT%": wt_percent[2],
            "Temperature_C": round(temperature, 2),
            "Humidity_pct": round(humidity, 2),
            "Pressure_atm": round(pressure, 3),
            "pH": round(ph_value, 2),
            "StirringSpeed_rpm": round(stirring_speed, 1),
            "ReactionTime_h": round(reaction_time, 2),
            "Target_Property": round(target, 3),
            "Sample_ID": f"MIX_DBG{i+1:04d}",
            "Total_WT%": wt_percent.sum(),  # 検証用（常に 100%）
            "Notes": "Synthetic mixture data with numeric features for debugging"
        })

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")


def generate_mixture_4comp_samples(n: int, filename: str, with_numeric: bool = False):
    """
    4成分の混合物データ生成：化合物 4 列（SMILES）、重量％分率（WT%）
    """
    compound_pool = [
        ("CCO", "Ethanol"),
        ("CC(C)O", "Isopropanol"),
        ("c1ccccc1", "Benzene"),
        ("CC1=CC=CC=C1", "Toluene"),
        ("CC(=O)Oc1ccccc1C(=O)O", "Aspirin"),
        ("CN1=C(C(N(C1=O)=O)C)N(C)C", "Caffeine"),  # SMILES 修正
        ("CCC1=CC(=C(C=C1)O)C=O", "Vanillin"),
        ("CC(C)Cc1ccc(cc1)C(C)C(=O)O", "Ibuprofen"),
        ("CC(=O)O", "Acetic Acid"),
        ("C1CCCCC1", "Cyclohexane"),
    ]

    data = []
    for i in range(n):
        num_comp = np.random.randint(2, 5)  # 2〜4成分
        indices = np.random.choice(len(compound_pool), size=num_comp, replace=False)
        raw_weights = np.random.uniform(10, 90, size=num_comp)
        wt_percent = (raw_weights / raw_weights.sum() * 100).round(2)

        row = {"Sample_ID": f"MIX4_{i + 1:04d}"}
        # 4列分用意（足りない分は空）
        for j in range(4):
            if j < num_comp:
                row[f"Compound_{j + 1}_Name"] = compound_pool[indices[j]][1]
                row[f"Compound_{j + 1}_SMILES"] = compound_pool[indices[j]][0]
                row[f"Compound_{j + 1}_WT%"] = wt_percent[j]
            else:
                row[f"Compound_{j + 1}_Name"] = ""
                row[f"Compound_{j + 1}_SMILES"] = ""
                row[f"Compound_{j + 1}_WT%"] = np.nan

        if with_numeric:
            row["Temperature_C"] = round(np.random.uniform(20, 100), 1)
            row["Pressure_atm"] = round(np.random.uniform(0.5, 5.0), 2)
            row["pH"] = round(np.random.uniform(2, 12), 2)
            row["StirringSpeed_rpm"] = np.random.randint(100, 1000)
            row["ReactionTime_h"] = round(np.random.uniform(1, 48), 1)
            # 収率ターゲット
            row["Target_Yield_pct"] = round(np.random.uniform(10, 100), 2)
        else:
            # 沸点ターゲット（模造）
            row["Target_BoilingPoint_C"] = round(np.random.uniform(50, 200), 1)

        data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")

# ========== 通常テーブルデータ生成 ==========
def generate_tabular_samples(n: int, filename: str, task: str = "regression"):
    if task == "regression":
        X, y = make_regression(
            n_samples=n, n_features=8, n_informative=5,
            noise=10, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]

        # カテゴリカル特徴量の追加
        df["Category"] = np.random.choice(["TypeA", "TypeB", "TypeC"], size=n)

        # 欠損値の人工的追加（5%）
        mask = np.random.random((n, 3)) < 0.05
        for col_idx in range(3):
            df.iloc[mask[:, col_idx], col_idx] = np.nan

    else:  # classification
        X, y = make_classification(
            n_samples=n, n_features=10, n_informative=7,
            n_redundant=2, n_classes=3, random_state=42
        )
        df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(X.shape[1])])
        df["Target_Class"] = y
        df["Sample_ID"] = [f"S{i+1:04d}" for i in range(n)]

    df.to_csv(filename, index=False, encoding="utf-8-sig")
    print(f"Generated {filename} ({len(df)} rows)")


def generate_debug_samples():
    """コンポーネントが期待するデバッグ用サンプル 5 種を生成"""
    debug_dir = "data/samples/debug"
    os.makedirs(debug_dir, exist_ok=True)

    # 1. 混合物回帰 (WT% + 数値)
    generate_mixture_samples_with_numeric(50, f"{debug_dir}/mixture_regression_debug.csv")

    # 1-2. 混合物 SMILES の回帰 (4成分)
    with open(f"{debug_dir}/mixture_smiles_only.csv", "w", encoding="utf-8-sig") as f:
        f.write(MIXTURE_SMILES_ONLY_DATA)
    print(f"Generated {debug_dir}/mixture_smiles_only.csv (Fixed)")

    # 1-3. 混合物 SMILES + 数値 (4成分)
    with open(f"{debug_dir}/mixture_smiles_numeric.csv", "w", encoding="utf-8-sig") as f:
        f.write(MIXTURE_SMILES_NUMERIC_DATA)
    print(f"Generated {debug_dir}/mixture_smiles_numeric.csv (Fixed)")

    # 2. 単調性制約テスト
    n = 100
    np.random.seed(42)
    mw = np.random.uniform(100, 500, n)
    logp = np.random.uniform(-1, 5, n)
    tpsa = np.random.uniform(20, 150, n)
    # 単調減少（MW, LogP, TPSAが上がると溶解度が下がる傾向）
    sol = 5 - (0.01 * mw) - (0.5 * logp) - (0.005 * tpsa) + np.random.normal(0, 0.2, n)
    df_mono = pd.DataFrame({
        "MW": mw.round(2),
        "LogP": logp.round(2),
        "TPSA": tpsa.round(2),
        "Solubility_mg_L": sol.round(3)
    })
    df_mono.to_csv(f"{debug_dir}/monotonicity_test.csv", index=False)
    print(f"Generated {debug_dir}/monotonicity_test.csv")

    # 3. 時系列リーク検出テスト
    n = 50
    dates = pd.date_range(start="2024-01-01", periods=n, freq="D")
    df_leak = pd.DataFrame({
        "Date": dates.strftime("%Y-%m-%d"),
        "Batch_ID": [f"BATCH_{i // 5:02d}" for i in range(n)],
        "Temperature": np.random.uniform(50, 100, n),
        "Pressure": np.random.uniform(1, 5, n),
        # 時系列に強く依存するターゲット（リークの元）
        "Yield_pct": (np.linspace(70, 95, n) + np.random.normal(0, 1, n)).round(2)
    })
    df_leak.to_csv(f"{debug_dir}/timeseries_leak_test.csv", index=False)
    print(f"Generated {debug_dir}/timeseries_leak_test.csv")

    # 4. xTB 外部ツール依存テスト (小分子)
    small_smiles = ["C", "CC", "CCC", "CCO", "CCN", "c1ccccc1", "C1CCCCC1", "O=C=O", "N", "O"]
    data_xtb = []
    for i in range(20):
        smi = small_smiles[i % len(small_smiles)]
        data_xtb.append({
            "SMILES": smi,
            "ID": f"MOL_{i:02d}",
            "HOMO_eV": np.nan  # xTBで計算すべき場所
        })
    df_xtb = pd.DataFrame(data_xtb)
    df_xtb.to_csv(f"{debug_dir}/xtb_dependency_test.csv", index=False)
    print(f"Generated {debug_dir}/xtb_dependency_test.csv")

    # 5. 分類タスク (バランス済み)
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, weights=[0.5, 0.5], random_state=42)
    df_cls = pd.DataFrame(X, columns=[f"Feature_{i}" for i in range(10)])
    df_cls["Activity"] = y
    df_cls.to_csv(f"{debug_dir}/classification_balanced.csv", index=False)
    print(f"Generated {debug_dir}/classification_balanced.csv")

# ========== 実行 ==========
if __name__ == "__main__":
    os.makedirs("data/samples", exist_ok=True)

    # SMILES データ
    generate_smiles_samples(25, "data/samples/smiles_25_quick.csv")
    generate_smiles_samples(100, "data/samples/smiles_100_ml.csv")
    generate_smiles_samples(500, "data/samples/smiles_500_stress.csv")

    # 通常テーブルデータ
    generate_tabular_samples(50, "data/samples/tabular_50_simple.csv", "regression")
    generate_tabular_samples(200, "data/samples/tabular_200_complex.csv", "regression")
    generate_tabular_samples(1000, "data/samples/tabular_1000_large.csv", "regression")

    # 混合物データ
    generate_mixture_samples(30, "data/samples/mixture_30_simple.csv")
    generate_mixture_samples(100, "data/samples/mixture_100_ml.csv")

    # 混合物データ（数値特徴量付き：デバッグ用）
    generate_mixture_samples_with_numeric(50, "data/samples/mixture_50_debug_numeric.csv")

    # デバッグ用追加セット
    generate_debug_samples()

    print("\nAll sample data generated successfully!")
