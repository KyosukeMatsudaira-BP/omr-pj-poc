from pathlib import Path
import json
from tqdm import tqdm
import re
import csv
import concurrent.futures
from functools import partial
import logging

# ログの設定
logging.basicConfig(level=logging.INFO)

def extract_rel_position(text):
    # 正規表現で rel_position の値を抽出、負の数にも対応
    match = re.search(r'rel_position:(-?\d+)', text)
    if match:
        return match.group(1)
    return None


def task_for_each_label_json(
    label_filepath: str,
    labels_to_use: dict,
    label_results_dir: Path,
    barline_results_dir: Path,
):
    # ラベルデータの読み込み
    with open(label_filepath) as file:
        label_json = json.load(file)

    # アノテーションを取得
    annotations = label_json["annotations"]

    annotations_grouped_by_img_id = {}
    # 前処理：画像idをキーとする辞書を作成
    for k, v in annotations.items():
        img_id = v["img_id"]
        if img_id not in annotations_grouped_by_img_id:
            annotations_grouped_by_img_id[img_id] = []
        annotations_grouped_by_img_id[img_id].append(v)
    
    # 各画像idごとにアノテーションデータをファイルに保存
    for img_id, annotation_list in annotations_grouped_by_img_id.items():
        filename = f"{img_id}.txt"
        savefilepath = label_results_dir / filename

        with open(savefilepath, mode='w') as f:
            writer = csv.writer(f)
            for item in annotation_list:
                data = []
                cat_id_first = int(item["cat_id"][0])
                if cat_id_first in labels_to_use: # 使うラベル情報だけを書き込む 
                    a_bbox = item["a_bbox"]
                    comments = item["comments"]
                    rel_position = "_"
                    if "rel_position" in comments:
                        rel_position = extract_rel_position(comments)
                    data.append(cat_id_first)
                    data.append(rel_position)
                    data.extend(a_bbox)
                    # 書き込み
                    writer.writerow(data)
            # 小節線のデータを書き込み
            # 小節線のアノテーションデータを読み込み
            barline_result_filepath = barline_results_dir / f"{img_id}.txt"
            with open(barline_result_filepath, mode="r") as f:
                reader = csv.reader(f)
                label_id = 209
                rel_position = "_"
                for item in reader:
                    img_id = item[0]
                    v = item[1:]
                    data = []
                    data.append(label_id)
                    data.append(rel_position)
                    data.extend(v)
                    writer.writerow(data)
    return f"{Path(label_filepath).name} has completed."


if __name__=="__main__":
    # ディレクトリパスの定義
    base_dir = Path(__file__).parent.parent
    data_version = "ds2_dense" # フル版ならds2_complete
    data_dir = base_dir / "data" / data_version
    results_dir = base_dir / "results"
    label_results_dir = results_dir / "label" / data_version
    barline_results_dir = results_dir / "barline_annotation" / data_version / "results"
    label_results_dir.mkdir(exist_ok=True, parents=True)

    # 使用するラベルのリスト
    labels_to_use = {
        6: "clefG",
        7: "clefCAlto",
        8: "clefCTenor",
        9: "clefF",
        25: "noteheadBlackOnLine",
        26: "noteheadBlackOnLineSmall",
        27: "noteheadBlackInSpace",
        28: "noteheadBlackInSpaceSmall",
        29: "noteheadHalfOnLine",
        30: "noteheadHalfOnLineSmall",
        31: "noteheadHalfInSpace",
        32: "noteheadHalfInSpaceSmall",
        33: "noteheadWholeOnLine",
        34: "noteheadWholeOnLineSmall",
        35: "noteheadWholeInSpace",
        36: "noteheadWholeInSpaceSmall",
        37: "noteheadDoubleWholeOnLine",
        38: "noteheadDoubleWholeOnLineSmall",
        39: "noteheadDoubleWholeInSpace",
        40: "noteheadDoubleWholeInSpaceSmall",
        60: "accidentalFlat",
        61: "accidentalFlatSmall",
        62: "accidentalNatural",
        63: "accidentalNaturalSmall",
        64: "accidentalSharp",
        65: "accidentalSharpSmall",
        66: "accidentalDoubleSharp",
        67: "accidentalDoubleFlat",
        68: "keyFlat",
        69: "keyNatural",
        70: "keySharp",
        100: "graceNoteAcciaccaturaStemUp",
        101: "graceNoteAppoggiaturaStemUp",
        102: "graceNoteAcciaccaturaStemDown",
        103: "graceNoteAppoggiaturaStemDown",
        135: "staff",
        209: "barLine", # 208までdeepscoreにラベルが存在するため209に設定
    }

    # jsonごとに並列実行してデータを作成する
    partial_task = partial( # partialを使って共通の引数を固定する
        task_for_each_label_json,
        labels_to_use=labels_to_use,
        label_results_dir=label_results_dir,
        barline_results_dir=barline_results_dir
    )
    label_filepaths = [str(file) for file in data_dir.rglob("*.json")]
    print(label_filepaths)
    with concurrent.futures.ThreadPoolExecutor(max_workers=13) as executor:
        # 各タスクをサブミットしてfutureオブジェクトを生成
        futures = [executor.submit(partial_task, label_filepath) for label_filepath in label_filepaths]
        
        # タスクが完了するたびにログを出力
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            logging.info(result)




