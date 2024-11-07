import pytest
import torch
from src.domain.model import OMRModel
from yolov3.models.yolo import Model
from src.domain.dataloader import CustomDataset, custom_collate_fn
from torch.utils.data import DataLoader
from tests.utils import get_test_data_path

@pytest.fixture()
def test_model_fixture():
    config_path = 'models/config/omr_yolov5s.yaml' #'yolov3/models/yolov5s.yaml'
    model_path = 'models/pre_trained/yolov5s.pt'
    model = OMRModel(config_path)
    model.load_state_dict(torch.load(model_path)['model'].state_dict())  # yolov5s.ptは、学習済みの重みファイル
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    return model, device

@pytest.fixture()
def test_data_fixture():
    img_dir = get_test_data_path(__file__, "images")
    annotation_dir = get_test_data_path(__file__, "labels")
    train_dataset = CustomDataset(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
    )
    return DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)

def test_omr_model(test_model_fixture, test_data_fixture):
    """OMRModelのテスト
    - モデルの出力のshapeが正しいか
    """
    model, device = test_model_fixture
    data_loader = test_data_fixture

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)

        for i in range(len(outputs)):
            assert outputs[i].shape[0] == 1
            assert outputs[i].shape[1] == 3
            assert outputs[i].shape[2] == 72/2**i
            assert outputs[i].shape[3] == 72/2**i
            assert outputs[i].shape[4] == 85
