# Aryth Key Finder v0.4 Beta — Streamlit Edition

楽曲の**主調、Camelot表記、複数回の転調、元キーへの復帰**を
自動解析するWebアプリです。

## 表記

- `G Major（長調） = 9B`
- `E Minor（短調） = 9A`
- Camelotの **B＝Major（長調）**
- Camelotの **A＝Minor（短調）**

アプリ内に全24キーの対応表があります。

## 主な機能

- MP3 / WAV / M4A / FLAC
- Major / Minorを含むキー判定
- Camelot表記
- 複数回の転調検出
- 元キー・過去のキーへの復帰検出
- 転調位置と推定移調量
- ♯優先 / ♭優先
- 転調感度3段階
- 解析結果をセッション内に保持
- 一時音源を解析後に削除

## ファイル構成

```text
streamlit_app.py       Streamlit UI
key_analyzer.py        解析エンジン
requirements.txt       Python依存関係
packages.txt           FFmpeg
.streamlit/config.toml Streamlit設定
DEPLOY.md              公開手順
```

## ローカル実行

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run streamlit_app.py
```

Windows PowerShell：

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## 注意

自動判定のため、必ず正解するわけではありません。

- 属調・下属調方向のコード進行を転調として検出する場合があります。
- 相対長調と相対短調を混同する場合があります。
- 借用和音やモード変化を転調と判断する場合があります。
- コード感の薄い曲やドラム中心の曲では精度が下がります。

結果は耳、DAW、楽譜やコード解析と併用してください。

## ライセンス

MIT License
