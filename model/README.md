## 環境構築手順
1. gitからyolov3をクローン
    ```
    git clone https://github.com/ultralytics/yolov3.git
    ```
1. dockerコンテナ作成
    ```
    docker compose up -d
    docker exec -it {コンテナ名} bash
    ```
1. コンソールで実行する場合```poetry shell```

1. cocoデータセットの取得

    yolov3/data/scripts/get_coco128.shを以下のように変更
    ```
    d='../datasets' -> d='./data'
    ```
    以下を実行
    ```
    bash yolov3/data/scripts/get_coco128.sh
    ```
