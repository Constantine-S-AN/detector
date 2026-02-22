# Contributing

## Development Setup

```bash
make setup
```

## Quality Gates

```bash
make lint
make test
make demo
```

## Pull Requests

- 保持改动聚焦并附上复现实验命令。
- 如果修改 detector 或 feature，请更新测试。
- 如果修改前端 demo 数据结构，请同步更新 `scripts/export_demo_assets.py` 与页面读取逻辑。
