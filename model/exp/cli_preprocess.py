import logging
import logging.config
import os
import sys

import click
import cv2
from tqdm import tqdm

sys.path.append(
    "yolov3"
)  # TODO: 【暫定対応】yolov3内の参照がyolov3以下からのため、明示的にyolov3をパスに追加している。必要なモジュールは用意してyoloは利用しないようにする

logger = logging.getLogger()

logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


@click.group()
def cli():
    pass


# クラスマッピングを定義
class_mapping = {
    "_": 0,  # 音符以外の記号
    "-12": 1,  # 音符（C）
    "-11": 2,  # 音符（D）
    "-10": 3,  # 音符（E）
    "-9": 4,  # 音符（F）
    "-8": 5,  # 音符（G）
    "-7": 6,  # 音符（A）
    "-6": 7,  # 音符（B）
    "-5": 8,  # 音符（C）
    "-4": 9,  # 音符（D）
    "-3": 10,  # 音符（E）
    "-2": 11,  # 音符（F）
    "-1": 12,  # 音符（G）
    "0": 13,  # 音符（A）
    "1": 14,  # 音符（B）
    "2": 15,  # 音符（C）
    "3": 16,  # 音符（D）
    "4": 17,  # 音符（E）
    "5": 18,  # 音符（F）
    "6": 19,  # 音符（G）
    "7": 20,  # 音符（A）
    "8": 21,  # 音符（B）
    "9": 22,  # 音符（C）
    "10": 23,  # 音符（D）
    "11": 24,  # 音符（E）
    "12": 25,  # 音符（F）
}


def load_image(image_path):
    """画像を読み込んでnumpy配列に変換する関数

    Args:
        image_path (str): 画像ファイルのパス(.pngまたは.jpg)

    Returns:
        np.ndarray: 読み込んだ画像のnumpy配列 (W, H, C)
    """
    # 画像の読み込み
    img = cv2.imread(image_path)

    # BGRからRGBに変換
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img.transpose(1, 0, 2)


def modify_line(line: str, img_shape: list, class_mapping: dict):
    """各行のアノテーションを変更する関数

    Args:
        line (str): アノテーションの1行
        img_shape (list): 画像のサイズ (W, H)

    Returns:
        str: 変更後のアノテーションの1行
    """
    values = line.strip().split(",")
    # ピッチクラスラベルのマッピング
    values = modify_class_labels(values, class_mapping)
    # whを計算
    values = modify_wh(values)
    # xy座標を中心に変換
    values = modify_xy_to_center(values)
    # xywhを正規化
    values = rescale_xywh(values, img_shape)
    return " ".join(values) + "\n"


def rescale_xywh(values: list, img_shpae: list):
    """bbboxのxywhを正規化する関数

    Args:
        values (list): アノテーションの1行のリスト
        img_shpae (list): 画像のサイズ (W, H)

    Returns:
        list: 正規化後のアノテーションの1行のリスト
    """
    for i in range(4):
        values[i + 2] = f"{float(values[i + 2]) / img_shpae[(i + 2) % 2]:.4f}"
    # if float(values[4]) > 1:
    #     print(values)
    #     assert float(values[4]) <= 1

    return values


def modify_class_labels(values: list, class_mapping: dict):
    """各行のアノテーションのクラスラベルを変更する関数

    Args:
        values (list): アノテーションの1行のリスト
        class_mapping (dict): 変更前と変更後のクラスラベルの対応辞書
    """
    # クラスラベルを変更（マッピングに存在する場合のみ）
    old_label = values[1]
    if old_label in class_mapping:
        values[1] = str(class_mapping[old_label])
    else:
        values[1] = "0"
    return values


def modify_wh(values: list):
    """各行のアノテーションのwhを計算する関数"""
    values[4] = f"{float(values[4]) - float(values[2]):.4f}"
    values[5] = f"{float(values[5]) - float(values[3]):.4f}"
    return values


def modify_xy_to_center(values: list):
    """各行のアノテーションのxy座標を中心に変換する関数"""

    # modfiy_whの処理済みのため、4,5はwhとなっている
    values[2] = f"{float(values[2]) + float(values[4]) / 2:.4f}"
    values[3] = f"{float(values[3]) + float(values[5]) / 2:.4f}"
    return values


@cli.command()
def preprocessing():
    # アノテーションディレクトリのパスを指定して実行
    data_dir = "data/complete/"
    save_dir = "data/complete/labels_mapping_to_center_and_normalization/"

    os.makedirs(save_dir, exist_ok=True)

    annotation_dir = data_dir + "label/"
    image_dir = data_dir + "images/"
    # ディレクトリ内の全てのtxtファイルを処理

    for filename_image in tqdm(os.listdir(image_dir)):
        try:
            if filename_image.endswith(".png"):
                image_path = os.path.join(image_dir, filename_image)
                image = load_image(image_path)

                filename_label = filename_image.replace(".png", ".txt")
                annotation_path = os.path.join(annotation_dir, filename_label)
                with open(annotation_path, "r") as file:
                    lines = file.readlines()

                save_file_path = os.path.join(save_dir, filename_label)
                # 修正された内容を書き込む
                with open(save_file_path, "w") as file:
                    try:
                        lines = [
                            line
                            for line in lines
                            if line.split(",")[0] not in ["135", "209"]
                        ]
                        modified_lines = [
                            modify_line(line, list(image.shape), class_mapping)
                            for line in lines
                        ]
                    except AssertionError as e:
                        print(filename_image)
                        raise e

                    file.writelines(modified_lines)
        except Exception:
            print(filename_image)


if __name__ == "__main__":
    cli()
