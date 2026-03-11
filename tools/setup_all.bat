@echo off
REM ============================================================
REM setup_all.bat — chemai2 開発環境セットアップスクリプト
REM 初回セットアップ or 新PC移行時に実行してください
REM ============================================================

echo.
echo ╔══════════════════════════════════════════════════════════╗
echo ║   chemai2 開発環境セットアップ                           ║
echo ╚══════════════════════════════════════════════════════════╝
echo.

REM スクリプトのディレクトリ = tools/
set TOOLS_DIR=%~dp0
set ROOT_DIR=%TOOLS_DIR%..
set XTB_BIN=%TOOLS_DIR%xtb-6.7.1\bin

echo [1/5] Python パッケージをインストールしています...
echo.
pip install -r "%ROOT_DIR%\requirements.txt"

echo.
echo [2/5] オプショナルパッケージをインストールしています...
echo      (mordred, unipka, ase, openCOSMO-RS)
echo.
pip install mordred
pip install unipka
pip install ase
pip install git+https://github.com/TUHH-TVT/openCOSMO-RS_py.git

echo.
echo [3/5] mlxtend をインストールしています... (Excluder用)
pip install mlxtend

echo.
echo [4/5] XTB バイナリ PATH を設定しています...
echo      XTB バイナリ場所: %XTB_BIN%
if exist "%XTB_BIN%\xtb.exe" (
    echo      xtb.exe を検出しました。
    REM ユーザー環境変数に追加（恒久的）
    for /f "tokens=2*" %%A in ('reg query "HKCU\Environment" /v PATH 2^>nul') do set CURRENT_PATH=%%B
    echo %CURRENT_PATH% | findstr /i "%XTB_BIN%" > nul
    if errorlevel 1 (
        setx PATH "%CURRENT_PATH%;%XTB_BIN%"
        echo      PATH に XTB を追加しました。新しいターミナルで有効になります。
    ) else (
        echo      XTB はすでに PATH に含まれています。
    )
) else (
    echo      [警告] xtb.exe が見つかりません: %XTB_BIN%
    echo      https://github.com/grimme-lab/xtb/releases から
    echo      xtb-6.7.1-windows-x86_64.zip をダウンロードして
    echo      tools\xtb-6.7.1\ に解凍してください。
)

echo.
echo [5/5] 動作確認...
python -c "import rdkit; print('  RDKit:', rdkit.__version__)"
python -c "import mordred; print('  Mordred: OK')" 2>nul || echo   Mordred: 未インストール
python -c "import unipka; print('  UniPKa: OK')" 2>nul || echo   UniPKa: 未インストール
python -c "import ase; print('  ASE:', __import__('ase').__version__)" 2>nul || echo   ASE: 未インストール
where xtb >nul 2>&1 && echo   XTB: OK || echo   XTB: PATH 未設定

echo.
echo ============================================================
echo セットアップ完了！以下でアプリを起動できます:
echo   cd frontend_streamlit
echo   python -m streamlit run app.py
echo ============================================================
echo.
pause
