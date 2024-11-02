import os

def get_test_data_path(dir_path: str, file_path) -> str:
    """テストデータの相対的なファイルパスを取得.

    Args:
        dir_path (str): __file__ を想定
        file_path (_type_): dir_pathからみたファイルパスを想定

    Returns:
        str: 組み立てられたファイルパス
    """
    base_dir = os.path.dirname(dir_path)
    test_data_path = os.path.join(base_dir, file_path)

    assert os.path.exists(test_data_path)
    return test_data_path