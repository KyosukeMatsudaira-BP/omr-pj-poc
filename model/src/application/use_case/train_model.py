from typing import Optional

import torch
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from src.domain.dataloader import CustomDataset, custom_collate_fn
from src.domain.loss import CustomLoss
from src.domain.model import OMRModel
from src.shared.const import Const


class TrainModel:
    def __init__(
        self,
        config_path: str,
        model_path: str,
        hyp_path: str,
        img_dir: str,
        annotation_dir: str,
        model_output_dir: str,
        epochs: int,
        batch_size: int,
        transform: Optional[transforms.Compose] = None,
    ):
        self.config_path = config_path
        self.model_path = model_path
        self.hyp_path = hyp_path
        self.img_dir = img_dir
        self.annotation_dir = annotation_dir
        self.model_output_dir = model_output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs = epochs
        self.batch_size = batch_size
        self.transform = transform

    def train(self):
        model = self._load_model()
        train_loader = self._load_data(self.batch_size)
        criterion, optimizer, scheduler = self._load_loss(model)

        for epoch in range(self.epochs):
            model.train()
            running_loss = 0.0
            pbar = tqdm(
                train_loader, total=len(train_loader), bar_format=Const.TQDM_BAR_FORMAT
            )
            i = 0
            for images, targets in pbar:
                i += 1

                images, targets = images.to(self.device), targets.to(self.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss, loss_items = criterion(outputs, targets)
                loss.backward()
                optimizer.step()

                loss_value = loss.item()
                running_loss += loss_value
                pbar.set_description(
                    f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / i}"
                )

                # GPUメモリの解放
                del images, targets, outputs, loss, loss_items
                torch.cuda.empty_cache()

            # scheduler.step()

            print(
                f"Epoch [{epoch + 1}/{self.epochs}], Loss: {running_loss / len(train_loader)}"
            )

        # トレーニング済みモデルの保存
        torch.save(model.state_dict(), self.model_output_dir)
        print("model save!")

    def _load_model(self) -> OMRModel:
        model = OMRModel(self.config_path)
        # 事前学習済みモデルの読み込み
        pretrained_dict = torch.load(self.model_path)["model"].state_dict()
        model_dict = model.state_dict()

        # 最終層以外のパラメータのみを適用
        pretrained_dict = {
            k: v
            for k, v in pretrained_dict.items()
            if k in model_dict and not k.startswith("model.24")
        }  # model.24は最終層
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.to(self.device)

        # ハイパラの設定
        with open(self.hyp_path, errors="ignore") as f:
            hyp = yaml.safe_load(f)
        model.hyp = hyp

        return model

    def _load_loss(
        self, model: OMRModel
    ) -> tuple[CustomLoss, optim.Optimizer, optim.lr_scheduler.CosineAnnealingLR]:
        # カスタム損失関数
        criterion = CustomLoss(model)

        # オプティマイザとスケジューラー
        # YOLOv5の推奨学習率は0.01で、徐々に下げていく
        # 初期学習率を0.01に設定し、スケジューラーで調整する
        optimizer = optim.Adam(model.parameters(), lr=0.01)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.epochs,  # エポック数に合わせて学習率を調整
            eta_min=1e-6,  # 最小学習率
        )

        return criterion, optimizer, scheduler

    def _load_data(self, batch_size: int) -> DataLoader:
        # データローダ
        train_dataset = CustomDataset(
            img_dir=self.img_dir,
            annotation_dir=self.annotation_dir,
            transform=self.transform,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=custom_collate_fn,
        )

        return train_loader
