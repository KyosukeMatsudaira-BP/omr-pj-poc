import pytest
from src.domain.dataloader import CustomDataset
from tests.utils import get_test_data_path

@pytest.fixture()
def test_fixture():
    pass

def test_custom_dataset():
    """CustomDatasetのテスト
    - 画像二つを読み込み二つのデータセットを作成できているか
    - データセットの画像のサイズが正しいか
    """
    dataset = CustomDataset(
        img_dir=get_test_data_path(__file__, "images"),
        annotation_dir=get_test_data_path(__file__, "labels")
    )
    assert len(dataset) == 2
    assert dataset[0][0].shape == (3, 576, 576)