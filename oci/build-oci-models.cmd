@echo off
setlocal enabledelayedexpansion

:: Determine script location
set SCRIPT_DIR=%~dp0

:: Read secrets from .env file in the oci directory
echo Loading secrets from .env file...
set ENV_FILE=%SCRIPT_DIR%.env
if not exist "%ENV_FILE%" (
    echo ERROR: .env file not found at %ENV_FILE%
    echo Please create .env file with the following variables:
    echo DOCKER_PRIVATE=docker.cfg.deloitte.com
    echo DOCKER_SEMOSS=docker.semoss.org
    echo DOCKER_USER=robot_gitlab-pusher
    echo DOCKER_PASS=your_password
    echo SEMOSS_DOCKER_USER=your_username
    echo SEMOSS_DOCKER_PASS=your_password
    exit /b 1
)

for /f "tokens=1,2 delims==" %%a in (%ENV_FILE%) do (
    set %%a=%%b
)

:: Get current date/time for tagging
for /f "tokens=2 delims==" %%I in ('wmic os get localdatetime /format:list') do set datetime=%%I
set DATE_TAG=%datetime:~0,4%-%datetime:~4,2%-%datetime:~6,2%-%datetime:~8,4%

:: Login to registries
echo Logging in to Docker registry %DOCKER_PRIVATE%...
docker login %DOCKER_PRIVATE% -u %DOCKER_USER% -p %DOCKER_PASS%

echo Logging in to Docker registry %DOCKER_SEMOSS%...
docker login %DOCKER_SEMOSS% -u %SEMOSS_DOCKER_USER% -p %SEMOSS_DOCKER_PASS%

:: Set OCI directory path
set OCI_DIR=%SCRIPT_DIR%

:: List of directories to exclude from model detection
set EXCLUDE_DIRS=.env .venv download.py pyproject.toml build-oci-models.cmd __pycache__ .env.example OCI_README.md Dockerfile

:: Select model to build (or build all)
if "%1"=="" (
    echo Available models:
    for /d %%D in (%OCI_DIR%*) do (
        set SKIP=0
        for %%E in (%EXCLUDE_DIRS%) do (
            if "%%~nxD"=="%%E" set SKIP=1
        )
        if !SKIP!==0 echo - %%~nxD
    )
    set /p MODEL="Enter model name (or 'all' for all models): "
) else (
    set MODEL=%1
)

:: Build and push selected model(s)
if "%MODEL%"=="all" (
    for /d %%D in (%OCI_DIR%*) do (
        set SKIP=0
        for %%E in (%EXCLUDE_DIRS%) do (
            if "%%~nxD"=="%%E" set SKIP=1
        )
        if !SKIP!==0 call :build_model "%%~nxD"
    )
) else (
    if exist "%OCI_DIR%%MODEL%" (
        call :build_model %MODEL%
    ) else (
        echo Model %MODEL% not found in oci directory!
        exit /b 1
    )
)

echo All builds completed!
exit /b 0

:build_model
set MODEL_NAME=%1
echo Building model: %MODEL_NAME%

:: Create image names with organized structure
set PRIVATE_IMAGE=%DOCKER_PRIVATE%/genai/cfg-ms-models/oci/%MODEL_NAME%:%DATE_TAG%
set PRIVATE_LATEST=%DOCKER_PRIVATE%/genai/cfg-ms-models/oci/%MODEL_NAME%:latest
set SEMOSS_IMAGE=%DOCKER_SEMOSS%/genai/cfg-ms-models/oci/%MODEL_NAME%:%DATE_TAG%
set SEMOSS_LATEST=%DOCKER_SEMOSS%/genai/cfg-ms-models/oci/%MODEL_NAME%:latest

:: Build image using the shared Dockerfile with build args
echo Building Docker image for %MODEL_NAME%...
docker build ^
  --build-arg MODEL_PATH=%MODEL_NAME% ^
  --build-arg MODEL_NAME=%MODEL_NAME% ^
  -t %PRIVATE_IMAGE% ^
  -f %OCI_DIR%Dockerfile ^
  %OCI_DIR%

:: Tag all the variants
docker tag %PRIVATE_IMAGE% %PRIVATE_LATEST%
docker tag %PRIVATE_IMAGE% %SEMOSS_IMAGE%
docker tag %PRIVATE_IMAGE% %SEMOSS_LATEST%

:: Push all images
echo Pushing images for %MODEL_NAME%...
docker push %PRIVATE_IMAGE%
docker push %PRIVATE_LATEST%
docker push %SEMOSS_IMAGE%
docker push %SEMOSS_LATEST%

echo %MODEL_NAME% build and push completed successfully!
exit /b 0