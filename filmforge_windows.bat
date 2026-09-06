@echo off
setlocal
title FilmForge - one-shot setup + run (Lightning / Modal / Beam)
cd /d "%~dp0"

echo.
echo ==================================================
echo    FilmForge : setup + run
echo    Providers: Lightning AI  -^>  Modal  -^>  Beam
echo ==================================================
echo.

REM ---------------------------------------------------------------
REM 0. required files must sit next to this script
REM ---------------------------------------------------------------
if not exist "filmforge.py" (
  echo [X] filmforge.py not found in this folder:
  echo     %cd%
  echo     Put this .bat next to filmforge.py, modal_backend.py and the
  echo     lightning_*.* scripts, then re-run.
  pause & exit /b 1
)

REM ---------------------------------------------------------------
REM 1. Python present?
REM ---------------------------------------------------------------
where python >nul 2>nul
if errorlevel 1 (
  echo [X] Python not found. Install it, then reopen this window:
  echo     winget install -e --id Python.Python.3.11
  pause & exit /b 1
)

REM ---------------------------------------------------------------
REM 2. ffmpeg present? [needed to stitch clips]
REM ---------------------------------------------------------------
where ffmpeg >nul 2>nul
if errorlevel 1 (
  echo [!] ffmpeg not found - clip stitching will fail. Install with:
  echo     winget install -e --id Gyan.FFmpeg
  echo     Continuing...
)

REM ---------------------------------------------------------------
REM 3. isolated venv + python deps
REM ---------------------------------------------------------------
if not exist ".venv\Scripts\python.exe" (
  echo [*] Creating virtual environment .venv ...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo [*] Installing / upgrading Python packages ...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet --upgrade modal comfy-cli requests

REM ---------------------------------------------------------------
REM 4. make a sample shot list if none exists
REM ---------------------------------------------------------------
if not exist "shots.json" (
  echo [*] Writing a sample shots.json ...
  python -c "import json;open('shots.json','w').write(json.dumps({'fps':24,'width':768,'height':432,'shots':[{'id':'01','prompt':'wide establishing shot of a neon city at dawn, slow drift','seconds':2,'quality':'balanced'},{'id':'02','prompt':'close-up of the hero, shallow depth of field','seconds':2,'quality':'max'},{'id':'03','prompt':'the hero walks into frame as rain falls, dialogue','seconds':2,'quality':'draft'}],},indent=2))"
)

REM ---------------------------------------------------------------
REM 5. OFFLINE proof: build + stitch a film with placeholders
REM ---------------------------------------------------------------
echo.
echo [*] Running OFFLINE dry-run to verify the whole pipeline ...
python filmforge.py --shots shots.json --out film_dryrun.mp4 --transition crossfade --dry-run
echo.

REM ---------------------------------------------------------------
REM 6. ensure ComfyUI workflows exist (never clobbers your exports)
REM ---------------------------------------------------------------
echo [*] Ensuring ComfyUI workflows exist ...
python make_workflows.py

REM ---------------------------------------------------------------
REM 7. shared: pick model + Hugging Face token (used by all providers)
REM ---------------------------------------------------------------
if "%FF_MODEL%"=="" set "FF_MODEL=ltx23"
if "%HF_TOKEN%"=="" (
  echo.
  echo Paste a FRESH Hugging Face token [regenerate the old one first]:
  set /p HF_TOKEN=HF token: 
)

REM ---------------------------------------------------------------
REM 8. choose provider path
REM ---------------------------------------------------------------
echo.
echo Choose how to run your film:
echo    [1] Lightning AI Studio  - interactive ComfyUI over SSH (recommended)
echo    [2] Modal cloud          - auto-render the film (chains Modal -^> Beam)
echo    [3] Chain ALL            - try Lightning, fall back to Modal -^> Beam
echo.
set "CHOICE=1"
set /p CHOICE=Enter 1, 2 or 3 [default 1]: 
set "CHAIN=0"
if "%CHOICE%"=="3" set "CHAIN=1"
if "%CHOICE%"=="2" goto MODAL
goto LIGHTNING

REM ===============================================================
REM  PROVIDER 1 : LIGHTNING AI  (SSH-driven persistent Studio)
REM ===============================================================
:LIGHTNING
echo.
echo ==================================================
echo    PROVIDER 1/3 : Lightning AI Studio
echo ==================================================

where ssh >nul 2>nul
if errorlevel 1 (
  echo [!] OpenSSH client not found. Install it with:
  echo     winget install -e --id Microsoft.OpenSSH.Beta
  echo     [or: Settings ^> Apps ^> Optional Features ^> OpenSSH Client]
  goto LIGHTNING_FAIL
)

if "%LIGHTNING_SSH%"=="" (
  echo.
  echo In your Lightning Studio, click the SSH icon and copy the connection
  echo target - the part AFTER 'ssh', e.g.  s_01hxxxx@ssh.lightning.ai
  set /p LIGHTNING_SSH=Lightning SSH target: 
)
if "%LIGHTNING_SSH%"=="" (
  echo [!] No SSH target given.
  goto LIGHTNING_FAIL
)

echo [*] Copying setup scripts to the Studio ...
scp -o StrictHostKeyChecking=no lightning_setup.sh lightning_download.py lightning_go.sh "%LIGHTNING_SSH%:~/"
if errorlevel 1 goto LIGHTNING_FAIL

echo [*] Installing ComfyUI + custom nodes on the Studio [one-time] ...
ssh -o StrictHostKeyChecking=no "%LIGHTNING_SSH%" "bash -lc 'bash ~/lightning_setup.sh %FF_MODEL%'"
if errorlevel 1 goto LIGHTNING_FAIL

echo [*] Downloading model weights on the Studio ...
ssh -o StrictHostKeyChecking=no "%LIGHTNING_SSH%" "bash -lc 'source ~/.filmforge_env 2>/dev/null; export HF_TOKEN=%HF_TOKEN% && export HF_HUB_DISABLE_XET=1 && python ~/lightning_download.py %FF_MODEL%'"
if errorlevel 1 goto LIGHTNING_FAIL

echo [*] Launching ComfyUI on the Studio and tunneling port 8188 ...
start "FilmForge-ComfyUI-tunnel" cmd /c ssh -o StrictHostKeyChecking=no -L 8188:localhost:8188 "%LIGHTNING_SSH%" "bash -lc 'source ~/.filmforge_env 2>/dev/null; cd ~/ComfyUI && python main.py --listen 0.0.0.0 --port 8188'"
echo [*] Waiting ~25s for ComfyUI to boot ...
timeout /t 25 >nul
start "" http://localhost:8188
echo.
echo [OK] ComfyUI should now be open at  http://localhost:8188
echo     Render there: load a workflow, type your prompt, Queue Prompt.
echo     For LTX-2.3 grab the official graph:  python make_workflows.py --fetch-official
echo     Keep the tunnel window OPEN while rendering; close it + STOP the
echo     Studio when done so you don't burn free GPU hours.
goto END

:LIGHTNING_FAIL
echo [!] Lightning path could not run.
if "%CHAIN%"=="1" (
  echo     Chaining to the next provider: Modal ...
  goto MODAL
)
echo     Re-run and pick option 2 to use Modal instead.
goto END

REM ===============================================================
REM  PROVIDER 2 : MODAL  (headless auto-render; filmforge chains to Beam)
REM ===============================================================
:MODAL
echo.
echo ==================================================
echo    PROVIDER 2/3 : Modal cloud
echo ==================================================

if not exist "modal_backend.py" (
  echo [!] modal_backend.py not found - cannot run the Modal path.
  echo     Offline test is still done: film_dryrun.mp4
  goto END
)

echo [*] Modal auth [opens a browser the first time] ...
modal setup

echo [*] Creating Modal secret 'huggingface' ...
modal secret create huggingface HF_TOKEN=%HF_TOKEN% 2>nul || modal secret create huggingface HF_TOKEN=%HF_TOKEN% --force

echo [*] Downloading model weights into Modal Volumes [one-time, slow] ...
set "FF_GPU="
set "FF_MODEL=ltx23" & modal run modal_backend.py::download_models
set "FF_MODEL=wan22" & modal run modal_backend.py::download_models

echo [*] Deploying warm backends ...
set "FF_GPU=" & set "FF_MODEL=ltx23" & modal deploy modal_backend.py
set "FF_MODEL=wan22" & set "FF_GPU=A100-40GB" & modal deploy modal_backend.py
set "FF_GPU="

echo [*] Validating workflow node types against the deployed backends ...
set "FF_MODEL=ltx23" & modal run modal_backend.py::verify_workflow
set "FF_MODEL=wan22" & modal run modal_backend.py::verify_workflow

REM --- PROVIDER 3 : BEAM failover (optional). filmforge.py auto-uses it. ---
if "%BEAM_ENDPOINT%"=="" (
  echo.
  echo [3rd provider] Optional Beam failover. Leave BLANK to skip.
  set /p BEAM_ENDPOINT=Beam endpoint URL [optional]: 
)
if not "%BEAM_ENDPOINT%"=="" if "%BEAM_TOKEN%"=="" set /p BEAM_TOKEN=Beam token: 

echo [*] Generating your film [router chains Modal -^> Beam automatically] ...
python filmforge.py --shots shots.json --out film.mp4 --transition crossfade
echo.
echo [OK] Done. Output film: film.mp4
goto END

:END
echo.
pause
