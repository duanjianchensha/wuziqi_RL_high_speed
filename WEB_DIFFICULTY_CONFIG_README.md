# Web难度配置示例

## 配置说明

现在可以在 `gomoku/config.py` 中自定义Web界面的不同难度级别对应的MCTS模拟次数。

## 配置项

### 1. WEB_DIFFICULTY_MULTIPLIERS (倍数配置)
控制各难度相对于训练模拟次数的倍数：
```python
WEB_DIFFICULTY_MULTIPLIERS: dict = {
    "easy": 0.6,    # 简单：训练模拟次数的0.6倍
    "medium": 1.0,  # 中等：训练模拟次数的1.0倍
    "hard": 1.8     # 困难：训练模拟次数的1.8倍
}
```

### 2. WEB_DIFFICULTY_PLAYOUTS (绝对值配置)
直接指定各难度的绝对模拟次数（优先级高于倍数）：
```python
WEB_DIFFICULTY_PLAYOUTS: dict = {
    "easy": 200,    # 简单：固定200次模拟
    "medium": None, # None表示使用倍数计算
    "hard": 1000    # 困难：固定1000次模拟
}
```

## 使用示例

### 示例1：降低所有难度
```python
WEB_DIFFICULTY_MULTIPLIERS: dict = {
    "easy": 0.3,    # 更简单
    "medium": 0.6,  # 中等偏简单
    "hard": 1.0     # 困难相当于训练强度
}
```

### 示例2：固定模拟次数
```python
WEB_DIFFICULTY_PLAYOUTS: dict = {
    "easy": 100,    # 固定100次
    "medium": 300,  # 固定300次
    "hard": 800     # 固定800次
}
```

### 示例3：混合配置
```python
WEB_DIFFICULTY_MULTIPLIERS: dict = {
    "easy": 0.5,
    "medium": 1.0,
    "hard": 2.0
}
WEB_DIFFICULTY_PLAYOUTS: dict = {
    "easy": None,   # 使用倍数：0.5倍
    "medium": 400,  # 固定400次，覆盖倍数
    "hard": None    # 使用倍数：2.0倍
}
```

## 注意事项
- 修改配置后需要重启Web服务器（`python server.py`）
- 模拟次数过多会显著增加AI思考时间
- 系统会自动确保难度递增：easy ≤ medium ≤ hard