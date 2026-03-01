@echo off
:: scripts/build.bat
:: Phase 2 C++ 加速模块一键编译脚本（Windows MSVC）
:: 前提条件：
::   1. Visual Studio 2019/2022 已安装（含 C++ 桌面开发工作负载）
::   2. CMake >= 3.18 已安装并在 PATH 中
::   3. Python 已在 PATH 中
::   4. pybind11 已安装（pip install pybind11 或 git submodule）
:: 用法：双击运行 或 在项目根目录命令行执行 scripts\build.bat

setlocal EnableDelayedExpansion

echo ============================================
echo  Gomoku AlphaZero -- Phase 2 C++ 编译
echo ============================================

:: 检测 CMake
where cmake >nul 2>&1
if errorlevel 1 (
    echo [ERROR] 未找到 cmake，请安装 CMake 3.18+ 并加入 PATH
    pause & exit /b 1
)

:: 检测 pybind11（尝试两种方式）
set PYBIND11_FOUND=0
if exist "%~dp0..\extern\pybind11\CMakeLists.txt" (
    echo [INFO] 使用 extern/pybind11 submodule
    set PYBIND11_FOUND=1
) else (
    python -c "import pybind11" >nul 2>&1
    if not errorlevel 1 (
        echo [INFO] 使用 pip 安装的 pybind11
        set PYBIND11_FOUND=1
    )
)

if !PYBIND11_FOUND!==0 (
    echo [WARN] 未找到 pybind11，尝试自动安装...
    pip install pybind11
)

:: 切换到项目根目录
cd /d "%~dp0.."

:: 创建并进入 build 目录
if not exist build mkdir build
cd build

:: 配置 CMake（自动查找 VS 编译器）
echo.
echo [Step 1] CMake 配置...
cmake .. -G "Visual Studio 17 2022" -A x64
if errorlevel 1 (
    echo [RETRY] 尝试 VS 2019...
    cmake .. -G "Visual Studio 16 2019" -A x64
)
if errorlevel 1 (
    echo [ERROR] CMake 配置失败，请检查 Visual Studio 是否安装了 C++ 桌面开发工作负载
    cd ..
    pause & exit /b 1
)

:: 编译 Release
echo.
echo [Step 2] 编译 Release 版本...
cmake --build . --config Release --parallel
if errorlevel 1 (
    echo [ERROR] 编译失败
    cd ..
    pause & exit /b 1
)

cd ..

:: 验证生成的 .pyd 文件
for /f "tokens=*" %%f in ('dir /b /s gomoku_cpp*.pyd 2^>nul') do (
    echo.
    echo [SUCCESS] 编译成功: %%f
    echo 现在可以在训练和推理时使用 C++ 加速的 MCTS
    goto :done
)

echo [ERROR] 未找到生成的 gomoku_cpp*.pyd 文件
pause & exit /b 1

:done
pause
