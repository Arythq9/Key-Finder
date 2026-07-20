# Streamlit Community Cloud 公開手順

## 1. GitHubリポジトリを作る

1. GitHubへログイン
2. 「New repository」を開く
3. リポジトリ名を入力
4. PublicまたはPrivateを選択
5. リポジトリを作成

例：

```text
aryth-key-finder
```

## 2. ファイルをアップロード

このフォルダ内のファイルとフォルダを、
GitHubリポジトリの直下へアップロードします。

必須：

- `streamlit_app.py`
- `key_analyzer.py`
- `requirements.txt`
- `packages.txt`
- `.streamlit/config.toml`

公開説明用：

- `README.md`
- `LICENSE`
- `CHANGELOG.md`

ZIPファイルそのものではなく、解凍後の中身をアップロードしてください。

## 3. Streamlit Community Cloudへ接続

1. Streamlit Community CloudへGitHubアカウントでログイン
2. 「Create app」を押す
3. 「Yup, I have an app」を選ぶ
4. GitHubリポジトリとブランチを選ぶ
5. Main file pathへ次を入力

```text
streamlit_app.py
```

6. 任意のApp URLを設定
7. Deployを押す

## 4. ビルドを待つ

`requirements.txt`のPythonパッケージと、
`packages.txt`のFFmpegが自動でインストールされます。

アプリが起動したら、MP3またはWAVで動作を確認してください。

## 更新方法

GitHub上のファイルを更新してCommitすると、
Streamlit側も自動的に再デプロイされます。

## エラーが出た場合

### ModuleNotFoundError

`requirements.txt`がリポジトリ直下にあるか確認してください。

### 音声を読み込めない

- `packages.txt`に`ffmpeg`と1行だけ書かれているか確認
- 12秒以上20分以内の音源か確認
- 別のMP3またはWAVで確認

### メモリ上限

長い音源を同時に複数解析すると、
無料環境のメモリ上限へ達する可能性があります。

このアプリは1回につき1曲を処理します。
通常の3〜6分程度の楽曲から試してください。

### アプリが古い状態のまま

Streamlit Community Cloudの管理画面から
RebootまたはClear cacheを実行してください。
