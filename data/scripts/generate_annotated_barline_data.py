from pathlib import Path
import numpy as np
from scipy.spatial.distance import euclidean
from scipy.spatial import KDTree
from PIL import Image
import json
from tqdm import tqdm
import cv2


def get_files_with_extension(directory: Path, extension: str) -> list[str]:
    return list(directory.rglob(f'*{extension}'))

def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """ヘックスカラーコードをRGBに変換する関数"""
    hex_color = hex_color.lstrip('#')
    return tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

def get_img_id(annotation_data: dict, img_filename: str) -> str:
    """画像ファイル名から画像idを取得する"""
    img_id = [
        d["id"] for d in annotation_data if d["filename"]==img_filename
    ][0]
    return img_id


def find_color_pixels(image_array: np.ndarray, hex_color: str) -> np.ndarray:
    """指定した色のピクセルの座標を取得する"""
    target_color_rgb = hex_to_rgb(hex_color)
    # 画像がRGBAの場合、RGBだけに変換する
    if image_array.shape[2] == 4:  # 4チャンネル(RGBA)の場合
        image_array = image_array[:, :, :3]  # RGBの3チャンネルに切り取る

    # ターゲットカラーと一致するピクセルを見つける
    target_pixels = np.all(image_array == target_color_rgb, axis=-1)

    # 一致するピクセルの座標を取得
    coordinates = np.argwhere(target_pixels)
    return coordinates


def group_coordinates_by_distance(all_coordinates, threshold_distance):
    """
    すべての座標に対して距離がしきい値以下であれば同じグループ（オブジェクト）とみなす。
    KDTreeを使用して、効率的に近傍探索を行いグループ化します。

    Args:
        all_coordinates (list of tuples): 座標情報 [(x1, y1), (x2, y2), ...]。
        threshold_distance (float): グループ化するための距離のしきい値。

    Returns:
        list of list of tuples: グループ化された座標のリスト。
    """
    # 座標をnumpy配列に変換
    coordinates = np.array(all_coordinates)

    # KDTreeを作成
    tree = KDTree(coordinates)

    # 各座標のグループ化結果を格納するためのリスト
    groups = []
    
    # まだ処理されていない座標を追跡するフラグ
    unprocessed = set(range(len(coordinates)))

    # 各座標に対して探索
    while unprocessed:
        # まだ処理されていない最初の座標を取得
        index = unprocessed.pop()
        
        # 新しいグループを作成
        group = [index]
        
        # 近傍探索用のキューを用意
        queue = [index]
        
        while queue:
            current_index = queue.pop()
            # 現在の座標からしきい値以内の近傍点を探索
            neighbors = tree.query_ball_point(coordinates[current_index], threshold_distance)
            
            # 未処理の近傍点のみを追加
            for neighbor in neighbors:
                if neighbor in unprocessed:
                    unprocessed.remove(neighbor)
                    queue.append(neighbor)
                    group.append(neighbor)
        
        # グループを保存
        groups.append([all_coordinates[i] for i in group])
    
    return groups

def calculate_bounding_boxes_from_groups(groups):
    """
    グループ化された座標リストから、バウンディングボックスを計算します。
    
    Args:
        groups (list of list of tuples): グループ化された座標のリスト。

    Returns:
        list of tuples: 各オブジェクトのバウンディングボックス [(x, y, width, height), ...] のリスト。
    """
    bounding_boxes = []
    for group in groups:
        x_coordinates = [coord[0] for coord in group]
        y_coordinates = [coord[1] for coord in group]
        x_min, x_max = min(x_coordinates), max(x_coordinates)
        y_min, y_max = min(y_coordinates), max(y_coordinates)
        width = x_max - x_min
        height = y_max - y_min
        bounding_boxes.append([x_min, y_min, width, height])
    
    return bounding_boxes


def draw_bounding_boxes(image, bounding_boxes):
    """
    バウンディングボックスを画像に描画します。

    Args:
        image (numpy array): OpenCVで読み込んだ画像。
        bounding_boxes (list of tuples): バウンディングボックス [(x, y, width, height), ...] のリスト。
    
    Returns:
        numpy array: バウンディングボックスが描画された画像。
    """
    image_ = image.copy()
    for box in bounding_boxes:
        x, y, width, height = box
        cv2.rectangle(image_, (y, x), (y + height, x + width), (0, 0, 255), 2)
    
    return image_


if __name__=="__main__":
    base_dir = Path(__file__).parent.parent
    dense_data_dir = base_dir / "data" / "ds2_dense" 
    seg_dir = dense_data_dir/ "segmentation"
    results_dir = base_dir / "results" / "barline_annotation"
    save_img_dir = results_dir / "images"
    results_dir.mkdir(exist_ok=True)
    save_img_dir.mkdir(exist_ok=True)

    # dense版のアノテーションデータを読み込み
    train_json_file_path = dense_data_dir / "deepscores_train.json"
    test_json_file_path = dense_data_dir / "deepscores_test.json"
    with open(train_json_file_path) as file:
        deepscores_train_json = json.load(file)
    with open(test_json_file_path) as file:
        deepscores_test_json = json.load(file)
    dense_images_data = deepscores_train_json["images"] + deepscores_test_json["images"]

    # 元画像のパスを取得
    images_paths = get_files_with_extension(seg_dir, ".png")
    # images_paths = images_paths[:10]

    results = []
    for img_path in tqdm(images_paths):
        # 画像の読み込み
        img = Image.open(img_path).convert('RGB')
        img_np = np.array(img)
        
        # 小節線部分のピクセルの座標を取得する
        target_color_hex = "#00acc6" # 小節線の色
        barline_pixcel_coordinates = find_color_pixels(img_np, target_color_hex)

        # 距離に基づいて座標をグループ化
        grouped_coordinates = group_coordinates_by_distance(barline_pixcel_coordinates, threshold_distance=10)

        # グループ化された座標からバウンディングボックスを計算
        bounding_boxes = calculate_bounding_boxes_from_groups(grouped_coordinates)
        
        # 画像idを取得する
        try:
            img_filename = str(img_path.name).replace("_seg", "") # _segの文字列をファイル名から削除
            # print(img_filename)
            img_id = get_img_id(dense_images_data, img_filename)
        except:
            img_id = "None"
            img_filename = img_path.name
            print(img_filename)

        # 結果を追加
        for b_box in bounding_boxes:
            res = [img_id] + b_box
            results.append(res)

        # 画像の保存
        img_np = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY) # 白黒に一旦してから
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB) # RGBに変換することで三次元配列を維持
        img_with_bbox = draw_bounding_boxes(img_np, bounding_boxes) # bboxを描画
        img_result_filepath = save_img_dir / img_filename
        cv2.imwrite(img_result_filepath, img_with_bbox)
    
    # 結果をテキストファイルに出力
    result_path = results_dir / "barline_annotation.txt"
    with open(result_path, 'w') as f:
        for res in results:
            # リストの各要素をスペースで区切って1行に書き込む
            f.write(' '.join(map(str, res)) + '\n')
    




