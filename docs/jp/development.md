# 開発ガイド

## 開発環境のセットアップ

```bash
git clone https://github.com/Moge800/yolox_wrapper.git
cd yolox_wrapper
uv sync --group dev
```

---

## テストの実行

```bash
uv run pytest
```

カバレッジ付きで実行:

```bash
uv run pytest --cov=src/yolox_wrapper
```

### テスト構成

| ファイル | 対象 | 内容 |
|---------|------|------|
| `tests/test_config.py` | `config.py` | AppConfig の読み書き、プロファイル管理 |
| `tests/test_dataset.py` | `dataset.py` | DatasetPreparer の分割・リマップ処理 |
| `tests/test_wrapper.py` | `wrapper.py` | モデル初期化、レターボックス、NMS、推論結果 |

> `test_wrapper.py` は YOLOX パッケージ本体がなくてもテスト可能な範囲をカバーしています。

---

## リンター・フォーマッター

```bash
uv run ruff check .        # リントチェック
uv run ruff format .       # フォーマット
```

---

## プロジェクト構造

```
yolox_wrapper/
├── main.py                     # GUI エントリポイント
├── config.ini                  # アプリケーション設定
├── pyproject.toml              # パッケージ設定
├── src/
│   └── yolox_wrapper/
│       ├── __init__.py         # パッケージエクスポート
│       ├── config.py           # 設定管理 (AppConfig, ProfileParams)
│       ├── dataset.py          # データセット準備 (DatasetPreparer)
│       ├── wrapper.py          # メインラッパー (YOLOX, YOLOXResult, YOLOXBoxes)
│       ├── _trainer.py         # 内部学習エンジン (_YOLOXTrainer)
│       └── gui/
│           ├── __init__.py
│           ├── app.py          # メインウィンドウ (App)
│           ├── train_tab.py    # 学習タブ (TrainTab)
│           ├── infer_tab.py    # 推論タブ (InferTab)
│           ├── camera_tab.py   # カメラタブ (CameraTab)
│           └── export_tab.py   # エクスポートタブ (ExportTab)
├── tests/
│   ├── test_config.py
│   ├── test_dataset.py
│   └── test_wrapper.py
└── docs/                       # ドキュメント
```

---

## 依存関係の構造

```
wrapper.py ──→ _trainer.py ──→ dataset.py
    │
    └──→ dataset.py

config.py （独立）

gui/app.py ──→ gui/*_tab.py ──→ wrapper.py, config.py
```

---

## ライセンス

MIT License — 詳細は [LICENSE](../../LICENSE) を参照してください。
