from pathlib import Path
import json
from tqdm import tqdm
import re
import csv

def extract_rel_position(text):
    # 正規表現で rel_position の値を抽出、負の数にも対応
    match = re.search(r'rel_position:(-?\d+)', text)
    if match:
        return match.group(1)
    return None


if __name__=="__main__":
    # ディレクトリパスの定義
    base_dir = Path(__file__).parent.parent
    data_version = "ds2_dense" # フル版ならds2_complete
    data_dir = base_dir / "data" / data_version
    results_dir = base_dir / "results"
    label_results_dir = results_dir / "label" / data_version
    label_results_dir.mkdir(exist_ok=True)

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

    # 小節線のアノテーションデータを読み込み
    barline_results_dir = results_dir / "barline_annotation"
    barline_annotation_filepath = barline_results_dir / "barline_annotation.txt"
    barline_annotation = []
    with open(barline_annotation_filepath) as file:
        reader = csv.reader(file)
        for row in reader:
            barline_annotation.append([int(item) for item in row if item])
    barline_annotation_grouped_by_img_id = {}
    for item in barline_annotation:
        img_id = item[0]
        v = item[1:]
        if img_id not in barline_annotation_grouped_by_img_id:
            barline_annotation_grouped_by_img_id[img_id] = []
        barline_annotation_grouped_by_img_id[img_id].append(v)

    # 元データのラベルを読み込み
    ## ラベルファイルが多いので逐次読み込んでいく
    label_filepaths = [str(file) for file in data_dir.rglob("*.json")]
    for label_filepath in tqdm(label_filepaths):
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
        for img_id, annotation_list in tqdm(annotations_grouped_by_img_id.items()):
            filename = f"{img_id}.txt"
            savefilepath = label_results_dir / filename
            with open(savefilepath, 'w') as f:
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
                barline_list = barline_annotation_grouped_by_img_id.get(int(img_id), [])
                for item in barline_list:
                    data = []
                    label_id = 209
                    rel_position = "_"
                    data.append(label_id)
                    data.append(rel_position)
                    data.extend(item)
                    writer.writerow(data)




