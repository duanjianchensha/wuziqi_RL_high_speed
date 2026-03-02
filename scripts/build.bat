@echo off

chcp 65001 >nul

:: scripts/build.bat - Gomoku AlphaZero C++ build script (Windows MSVC)

setlocal EnableDelayedExpansion

echo ============================================

echo  Gomoku AlphaZero -- Phase 2 C++ Build

echo ============================================



:: ---------- 1. Find cmake (stop at FIRST match, newest VS first) ----------

where cmake >nul 2>&1

if not errorlevel 1 goto :cmake_ok

echo [INFO] cmake not in PATH, searching VS installations...

set CMAKE_FOUND=0

set VS_GEN_FROM_CMAKE=

for %%Y in (2022 2019 2017) do (

  if !CMAKE_FOUND!==0 (

    for %%E in (BuildTools Community Professional Enterprise) do (

      if !CMAKE_FOUND!==0 (

        for %%B in ("C:\Program Files (x86)" "C:\Program Files") do (

          if !CMAKE_FOUND!==0 (

            set "_TRY=%%~B\Microsoft Visual Studio\%%Y\%%E\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin"

            if exist "!_TRY!\cmake.exe" (

              set "PATH=!_TRY!;%PATH%"

              set CMAKE_FOUND=1

              if "%%Y"=="2022" set VS_GEN_FROM_CMAKE=Visual Studio 17 2022

              if "%%Y"=="2019" set VS_GEN_FROM_CMAKE=Visual Studio 16 2019

              if "%%Y"=="2017" set VS_GEN_FROM_CMAKE=Visual Studio 15 2017

              echo [INFO] Found cmake in VS%%Y/%%E  generator=!VS_GEN_FROM_CMAKE!

            )

          )

        )

      )

    )

  )

)

if !CMAKE_FOUND!==0 (

  echo [ERROR] cmake not found. Install CMake 3.18+ or VS with CMake component.

  pause & exit /b 1

)



:cmake_ok

cmake --version | findstr /i "cmake version"



:: ---------- 2. pybind11 ----------

set PYBIND11_CMAKE_DIR=

if exist "%~dp0..\extern\pybind11\CMakeLists.txt" (

    set "PYBIND11_CMAKE_DIR=%~dp0..\extern\pybind11"

    echo [INFO] Using extern/pybind11 submodule

) else (

    python -c "import pybind11" >nul 2>&1

    if errorlevel 1 (

        echo [WARN] pybind11 not found, installing...

        pip install pybind11

    )

    for /f "delims=" %%d in ('python -c "import pybind11; print(pybind11.get_cmake_dir())"') do (

        set "PYBIND11_CMAKE_DIR=%%d"

    )

    echo [INFO] pybind11 cmake dir: !PYBIND11_CMAKE_DIR!

)



:: ---------- 3. Determine VS generator ----------

if "!VS_GEN_FROM_CMAKE!"=="" (

    set VS_GENERATOR=

    for %%Y in (2022 2019 2017) do (

        if "!VS_GENERATOR!"=="" (

            for %%E in (BuildTools Community Professional Enterprise) do (

                if "!VS_GENERATOR!"=="" (

                    for %%B in ("C:\Program Files (x86)" "C:\Program Files") do (

                        if "!VS_GENERATOR!"=="" (

                            if exist "%%~B\Microsoft Visual Studio\%%Y\%%E\MSBuild\Current\Bin\MSBuild.exe" (

                                if "%%Y"=="2022" set VS_GENERATOR=Visual Studio 17 2022

                                if "%%Y"=="2019" set VS_GENERATOR=Visual Studio 16 2019

                                if "%%Y"=="2017" set VS_GENERATOR=Visual Studio 15 2017

                            )

                        )

                    )

                )

            )

        )

    )

    if "!VS_GENERATOR!"=="" (

        set VS_GENERATOR=Visual Studio 16 2019

        echo [WARN] Cannot detect VS version, defaulting to VS2019

    ) else (

        echo [INFO] Detected generator: !VS_GENERATOR!

    )

) else (

    set VS_GENERATOR=!VS_GEN_FROM_CMAKE!

)



:: ---------- 4. CMake configure ----------

cd /d "%~dp0.."

if not exist build mkdir build

cd build



:: Auto-clean stale generator cache

if exist CMakeCache.txt (

    findstr /i "CMAKE_GENERATOR:INTERNAL" CMakeCache.txt > "%TEMP%\gen_line.txt" 2>&1

    set /p CACHED_GEN=<"%TEMP%\gen_line.txt"

    echo !CACHED_GEN! | findstr /i "!VS_GENERATOR!" >nul 2>&1

    if errorlevel 1 (

        echo [INFO] Generator mismatch detected, cleaning build directory...

        cd ..

        rmdir /s /q build

        mkdir build

        cd build

    )

)



echo.

echo [Step 1] CMake configure with: !VS_GENERATOR!

if "!PYBIND11_CMAKE_DIR!"=="" (

    cmake .. -G "!VS_GENERATOR!" -A x64

) else (

    cmake .. -G "!VS_GENERATOR!" -A x64 -Dpybind11_DIR="!PYBIND11_CMAKE_DIR!"

)

if errorlevel 1 (

    echo [ERROR] CMake configure failed. Check VS C++ workload.

    cd ..

    pause & exit /b 1

)



:: ---------- 5. Build Release ----------

echo.

echo [Step 2] Building Release...

cmake --build . --config Release --parallel

if errorlevel 1 (

    echo [ERROR] Build failed.

    cd ..

    pause & exit /b 1

)

cd ..



:: ---------- 6. Verify .pyd ----------

for /f "tokens=*" %%f in ('dir /b /s gomoku_cpp*.pyd 2^>nul') do (

    echo.

    echo [SUCCESS] Build succeeded: %%f

    goto :done

)

echo [ERROR] gomoku_cpp*.pyd not found after build.

pause & exit /b 1



:done

pause