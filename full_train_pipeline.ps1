# full_train_pipeline.ps1
# 一键完整训练流水线：规则数据生成 → 监督预训练 → 自弈强化
#
# 用法：
#   .\full_train_pipeline.ps1
#   .\full_train_pipeline.ps1 -Games 10000 -Workers 4 -PretrainEpochs 100
#
# 阶段说明：
#   Phase 1  生成规则策略自弈数据（无 GPU，纯 CPU，多进程并行）
#   Phase 2  监督预训练（用规则数据快速传授战术知识）
#   Phase 3  AlphaZero 自弈强化（从高质量起点出发，规则数据持续 15% 混合）

param(
    [int]$Games         = 5000,   # 规则数据对局数
    [int]$Workers       = 3,      # 规则数据生成并行进程数
    [int]$PretrainEpochs = 80,    # 监督预训练 epoch 数
    [string]$RuleDataDir = "models/human_games/rule_data"
)

$ErrorActionPreference = "Stop"
$StartTime = Get-Date

function Write-Step([string]$msg) {
    Write-Host ""
    Write-Host ("=" * 60) -ForegroundColor Cyan
    Write-Host "  $msg" -ForegroundColor Cyan
    Write-Host ("=" * 60) -ForegroundColor Cyan
}

function Write-Done([string]$msg, [timespan]$elapsed) {
    Write-Host "  ✓ $msg  (耗时 $([math]::Round($elapsed.TotalMinutes,1)) 分钟)" -ForegroundColor Green
}

# ─────────────────────────────────────────────────────
Write-Step "Phase 1 / 3  生成规则策略对弈数据"
Write-Host "  对局数: $Games   并行 Worker: $Workers   输出目录: $RuleDataDir"
$t1 = Get-Date
python scripts/gen_rule_data.py `
    --games $Games `
    --workers $Workers `
    --out-dir $RuleDataDir
if ($LASTEXITCODE -ne 0) { Write-Error "规则数据生成失败，终止。"; exit 1 }
Write-Done "规则数据生成完成" ((Get-Date) - $t1)

# ─────────────────────────────────────────────────────
Write-Step "Phase 2 / 3  监督预训练（规则数据）"
Write-Host "  Epochs: $PretrainEpochs   数据目录: $RuleDataDir"
$t2 = Get-Date
python pretrain.py `
    --data-dir $RuleDataDir `
    --epochs $PretrainEpochs `
    --out models/current_policy.pth
if ($LASTEXITCODE -ne 0) { Write-Error "监督预训练失败，终止。"; exit 1 }
Write-Done "监督预训练完成 → models/current_policy.pth" ((Get-Date) - $t2)

# ─────────────────────────────────────────────────────
Write-Step "Phase 3 / 3  AlphaZero 自弈强化训练"
Write-Host "  从预训练模型起点开始，重置训练计数器"
Write-Host "  规则数据 (RULE_DATA_DIR) 在 batch 中持续 15% 混合采样"
$t3 = Get-Date
python train.py --fresh
if ($LASTEXITCODE -ne 0) { Write-Error "自弈训练异常退出。"; exit 1 }
Write-Done "自弈强化完成" ((Get-Date) - $t3)

# ─────────────────────────────────────────────────────
$total = (Get-Date) - $StartTime
Write-Host ""
Write-Host ("=" * 60) -ForegroundColor Yellow
Write-Host "  全流程完成！总耗时 $([math]::Round($total.TotalMinutes,1)) 分钟" -ForegroundColor Yellow
Write-Host "  最优模型: models/best_policy.pth" -ForegroundColor Yellow
Write-Host "  启动对战: python server.py" -ForegroundColor Yellow
Write-Host ("=" * 60) -ForegroundColor Yellow
