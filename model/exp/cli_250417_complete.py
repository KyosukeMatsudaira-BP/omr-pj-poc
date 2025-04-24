import logging
import logging.config
import os
import sys

import click
import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

sys.path.append(
    "yolov3"
)  # TODO: 【暫定対応】yolov3内の参照がyolov3以下からのため、明示的にyolov3をパスに追加している。必要なモジュールは用意してyoloは利用しないようにする
from src.domain.dataloader import CustomDataset, custom_collate_fn
from src.domain.loss import CustomLoss
from src.domain.model import OMRModel

logger = logging.getLogger()

logger.setLevel(logging.INFO)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)


BATCH_SIZE = 8
EPOCH_NUM = 2


@click.group()
def cli():
    pass


@cli.command()
@click.argument("exp-name", type=str)
def model_train(exp_name):
    """
    overview:
        学習済みモデルからファインチューニングしない方が、変な情報を保持せず良いかもしれないか確認
        音符は日常的な物体とはサイズ、形状が違うため、そう思った次第

    exp settings:
        size = 1280
        focal lossのgamma=5
        anchor boxを変更して実験
        学習済みモデルの利用をなし
    """
    model_dir = "models/"
    config_path = (
        model_dir + "config/omr_yolov5_change_anchor.yaml"
    )  #'yolov3/models/yolov5s.yaml'
    model_path = model_dir + "pre_trained/yolov5s.pt"

    data_dir = "data/"
    hyp_path = data_dir + "hyps/hyp.scratch-low.yaml"
    img_dir = data_dir + "complete/images/"
    annotation_dir = data_dir + "complete/labels_mapping_to_center_and_normalization/"

    fine_tune_dir = model_dir + f"fine_tuned/{exp_name}/"
    os.makedirs(
        fine_tune_dir,
        exist_ok=True,
    )
    output_path = fine_tune_dir + "omr_yolov5s.pth"

    model = OMRModel(config_path)
    # 事前学習済みモデルの読み込み
    pretrained_dict = torch.load(model_path)["model"].state_dict()
    model_dict = model.state_dict()

    # 最終層以外のパラメータのみを適用
    pretrained_dict = {
        k: v
        for k, v in pretrained_dict.items()
        if k in model_dict and not k.startswith("model.24")
    }  # model.24は最終層
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"device: {device}")
    model.to(device)

    # ハイパラの設定
    with open(hyp_path, errors="ignore") as f:
        hyp = yaml.safe_load(f)
    model.hyp = hyp

    # カスタム損失関数
    criterion = CustomLoss(model)

    # オプティマイザ
    optimizer = optim.Adam(model.parameters(), lr=0.002)

    # データローダ
    h_resize = 1280  # 32の倍数である必要がある
    w_resize = int(h_resize * 1)
    train_dataset = CustomDataset(
        img_dir=img_dir,
        annotation_dir=annotation_dir,
        transform=transforms.Compose(
            [
                transforms.Resize((h_resize, w_resize)),  # h,w
                transforms.ToTensor(),
            ]
        ),
    )
    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn
    )

    # tqdmの表示フォーマット
    TQDM_BAR_FORMAT = "{l_bar}{bar:10}{r_bar}"

    # トレーニングループ
    num_epochs = EPOCH_NUM

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, total=len(train_loader), bar_format=TQDM_BAR_FORMAT)
        i = 0
        for images, targets in pbar:
            i += 1

            images, targets = images.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss, loss_items = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            loss_value = loss.item()
            running_loss += loss_value
            pbar.set_description(
                f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / i}"
            )

            # GPUメモリの解放
            del images, targets, outputs, loss, loss_items
            torch.cuda.empty_cache()

        print(
            f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(train_loader)}"
        )

    # トレーニング済みモデルの保存
    torch.save(model.state_dict(), output_path)
    logger.info("model save!")


@cli.command()
@click.argument("exp-name", type=str)
def model_estim(exp_name):
    pass


if __name__ == "__main__":
    cli()
