import pytest
import yaml
import torch
from src.domain.model import OMRModel
from src.domain.dataloader import CustomDataset, custom_collate_fn
from src.domain.loss import CustomLoss
from torch.utils.data import DataLoader
from tests.utils import get_test_data_path

@pytest.fixture()
def fixture_model():
    config_path = 'models/config/omr_yolov5s.yaml' #'yolov3/models/yolov5s.yaml'
    model_path = 'models/pre_trained/yolov5s.pt'
    model = OMRModel(config_path)
    # model.load_state_dict(torch.load(model_path)['model'].state_dict())  # yolov5s.ptは、学習済みの重みファイル
    # 事前学習済みモデルの読み込み
    pretrained_dict = torch.load(model_path)['model'].state_dict()
    model_dict = model.state_dict()

    # 最終層以外のパラメータのみを適用
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                      if k in model_dict and not k.startswith('model.24')}  # model.24は最終層
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    device = 'cpu' # デバッグをする場合はgpuに渡すと中身が確認できないため
    model.to(device)

    hyp_path = "data/hyps/hyp.scratch-low.yaml"
    with open(hyp_path, errors="ignore") as f:
        hyp = yaml.safe_load(f)

    model.hyp = hyp

    return model, device

@pytest.fixture()
def fixture_data():
    img_dir = get_test_data_path(__file__, "images")
    annotation_dir = get_test_data_path(__file__, "labels")
    train_dataset = CustomDataset(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
    )
    return DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=custom_collate_fn)


def test_loss(fixture_model, fixture_data):

    model, device = fixture_model
    data_loader = fixture_data

    criterion = CustomLoss(model)

    for images, targets in data_loader:
        images, targets = images.to(device), targets.to(device)
        outputs = model(images)
        loss, _ = criterion(outputs, targets)

        assert loss.shape[0] == 1