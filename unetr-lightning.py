#!/usr/bin/env python3

# ! <<<
import sys
import pathlib
# ! >>>
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from monai.losses import DiceCELoss
from monai.inferers import sliding_window_inference
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    Compose,
    CropForegroundd,
    LoadImaged,
    Orientationd,
    RandFlipd,
    # ! <<<
    # RandCropByPosNegLabeld,
    Resized,
    # ! >>>
    RandShiftIntensityd,
    ScaleIntensityRanged,
    Spacingd,
    RandRotate90d,
    ToTensord,
)

from monai.config import print_config
from monai.metrics import DiceMetric
from monai.networks.nets import UNETR

from monai.data import (
    DataLoader,
    CacheDataset,
    load_decathlon_datalist,
    decollate_batch,
    list_data_collate,
    write_nifti
)

import torch
import pytorch_lightning
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
print_config()

# ## Setup data directory
#
# You can specify a directory with the `MONAI_DATA_DIRECTORY` environment variable.
# This allows you to save results and reuse downloads.
# If not specified a temporary directory will be used.

directory = os.environ.get("MONAI_DATA_DIRECTORY")
# ! <<<
# root_dir = tempfile.mkdtemp() if directory is None else directory
root_dir = '/home/yusongli/_dataset/_IIPL/ShuaiWang/20211223/unetr_output/'
# ! >>>
print(root_dir)

# ## Define the LightningModule (transform, network)
# The LightningModule contains a refactoring of your training code. The following module is a refactoring of the code in spleen_segmentation_3d.ipynb:


class Net(pytorch_lightning.LightningModule):
    def __init__(self):
        super().__init__()

        self._model = UNETR(
            in_channels=1,
            # ! <<<
            # out_channels=14,
            out_channels=2,
            # ! >>>
            img_size=(96, 96, 96),
            feature_size=16,
            hidden_size=768,
            mlp_dim=3072,
            num_heads=12,
            pos_embed="perceptron",
            norm_name="instance",
            res_block=True,
            conv_block=True,
            dropout_rate=0.0,
        ).to(device)

        self.loss_function = DiceCELoss(to_onehot_y=True, softmax=True)
        # ! <<<
        # self.post_pred = AsDiscrete(argmax=True, to_onehot=14)
        self.post_pred = AsDiscrete(argmax=True, to_onehot=2, num_classes=2)
        # self.post_label = AsDiscrete(to_onehot=14)
        self.post_label = AsDiscrete(to_onehot=2, num_classes=2)
        # ! >>>
        self.dice_metric = DiceMetric(
            include_background=False, reduction="mean", get_not_nans=False
        )
        self.best_val_dice = 0
        self.best_val_epoch = 0
        # ! <<<
        # self.max_epochs = 1300
        # self.check_val = 30
        self.max_epochs = 100
        self.check_val = 5
        # ! >>>
        self.warmup_epochs = 20
        self.metric_values = []
        self.epoch_loss_values = []

    def forward(self, x):
        return self._model(x)

    def prepare_data(self):
        # prepare data
        # data_dir ='/dataset/dataset0/'
        data_dir = 'dataset/'
        split_JSON = "dataset.json"
        datasets = data_dir + split_JSON
        datalist = load_decathlon_datalist(datasets, True, "training")
        val_files = load_decathlon_datalist(datasets, True, "validation")

        train_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Resized(keys=['image', 'label'], spatial_size=[96, 96, 96]),
                # ! <<<
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # Orientationd(keys=["image", "label"], axcodes="RAS"),
                # ! >>>
                ScaleIntensityRanged(
                    keys=["image"],
                    # ! <<<
                    # a_min=-175,
                    # a_max=250,
                    a_min=0,
                    a_max=1500,
                    # ! >>>
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # ! <<<
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                # RandCropByPosNegLabeld(
                #     keys=["image", "label"],
                #     label_key="label",
                #     spatial_size=(96, 96, 96),
                #     pos=1,
                #     neg=1,
                #     num_samples=4,
                #     image_key="image",
                #     image_threshold=0,
                # ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[0],
                #     prob=0.10,
                # ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[1],
                #     prob=0.10,
                # ),
                # RandFlipd(
                #     keys=["image", "label"],
                #     spatial_axis=[2],
                #     prob=0.10,
                # ),
                # RandRotate90d(
                #     keys=["image", "label"],
                #     prob=0.10,
                #     max_k=3,
                # ),
                # RandShiftIntensityd(
                #     keys=["image"],
                #     offsets=0.10,
                #     prob=0.50,
                # ),
                # ! >>>
                ToTensord(keys=["image", "label"]),
            ]
        )
        val_transforms = Compose(
            [
                LoadImaged(keys=["image", "label"]),
                AddChanneld(keys=["image", "label"]),
                Resized(keys=['image', 'label'], spatial_size=[96, 96, 96]),
                # ! <<<
                # Spacingd(
                #     keys=["image", "label"],
                #     pixdim=(1.5, 1.5, 2.0),
                #     mode=("bilinear", "nearest"),
                # ),
                # Orientationd(keys=["image", "label"], axcodes="RAS"),
                # ! >>>
                ScaleIntensityRanged(
                    keys=["image"],
                    # ! <<<
                    # a_min=-175,
                    # a_max=250,
                    a_min=0,
                    a_max=1500,
                    # ! >>>
                    b_min=0.0,
                    b_max=1.0,
                    clip=True,
                ),
                # ! <<<
                # CropForegroundd(keys=["image", "label"], source_key="image"),
                # ! >>>
                ToTensord(keys=["image", "label"]),
            ]
        )

        self.train_ds = CacheDataset(
            data=datalist,
            transform=train_transforms,
            cache_num=24,
            cache_rate=1.0,
            num_workers=8,
        )
        self.val_ds = CacheDataset(
            data=val_files,
            transform=val_transforms,
            cache_num=6,
            cache_rate=1.0,
            num_workers=8,
        )

    def train_dataloader(self):
        train_loader = torch.utils.data.DataLoader(
            self.train_ds,
            batch_size=1,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            collate_fn=list_data_collate,
        )
        return train_loader

    def val_dataloader(self):
        val_loader = torch.utils.data.DataLoader(
            self.val_ds,
            batch_size=1,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )
        return val_loader

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=1e-4, weight_decay=1e-5
        )
        return optimizer

    def training_step(self, batch, batch_idx):
        images, labels = (batch["image"].cuda(device),
                          batch["label"].cuda(device))
        output = self.forward(images)
        loss = self.loss_function(output, labels)
        tensorboard_logs = {"train_loss": loss.item()}
        return {"loss": loss, "log": tensorboard_logs}

    def training_epoch_end(self, outputs):
        avg_loss = torch.stack([x["loss"] for x in outputs]).mean()
        self.epoch_loss_values.append(avg_loss.detach().cpu().numpy())

    def validation_step(self, batch, batch_idx):
        images, labels = batch["image"], batch["label"]
        roi_size = (96, 96, 96)
        sw_batch_size = 4
        outputs = sliding_window_inference(
            images, roi_size, sw_batch_size, self.forward
        )
        loss = self.loss_function(outputs, labels)
        outputs = [self.post_pred(i) for i in decollate_batch(outputs)]
        labels = [self.post_label(i) for i in decollate_batch(labels)]
        # ! <<< Offline Predict
        img_number = pathlib.Path(batch['label_meta_dict']['filename_or_obj'][0]).parent.name
        save_folder = root_dir + str(self.current_epoch) + '/' + img_number + '/'
        
        save_folder_path = pathlib.Path(save_folder)
        save_folder_path.mkdir(parents=True, exist_ok=True)

        _, max_output = outputs[0].detach().cpu().max(axis=0, keepdim=False)
        max_output =  max_output.numpy()
        output_name =  save_folder + img_number + '_predict.nii.gz'
        write_nifti(max_output, output_name)

        _, max_label = labels[0].detach().cpu().max(axis=0, keepdim=False)
        max_label = max_label.numpy()
        label_name = save_folder + img_number + '_label.nii.gz'
        write_nifti(max_label, label_name)
        # ! >>>
        self.dice_metric(y_pred=outputs, y=labels)
        return {"val_loss": loss, "val_number": len(outputs)}

    def validation_epoch_end(self, outputs):
        val_loss, num_items = 0, 0
        for output in outputs:
            val_loss += output["val_loss"].sum().item()
            num_items += output["val_number"]
        mean_val_dice = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        mean_val_loss = torch.tensor(val_loss / num_items)
        tensorboard_logs = {
            "val_dice": mean_val_dice,
            "val_loss": mean_val_loss,
        }
        if mean_val_dice > self.best_val_dice:
            self.best_val_dice = mean_val_dice
            self.best_val_epoch = self.current_epoch
        print(
            f"current epoch: {self.current_epoch} "
            f"current mean dice: {mean_val_dice:.4f}"
            f"\nbest mean dice: {self.best_val_dice:.4f} "
            f"at epoch: {self.best_val_epoch}"
        )
        self.metric_values.append(mean_val_dice)
        return {"log": tensorboard_logs}

# ## Run the training


# initialise the LightningModule
net = Net()

# set up checkpoints
checkpoint_callback = ModelCheckpoint(
    dirpath=root_dir, filename="best_metric_model")

# initialise Lightning's trainer.
# ! <<<
trainer = pytorch_lightning.Trainer(
    gpus=[0],
    max_epochs=net.max_epochs,
    check_val_every_n_epoch=net.check_val,
    callbacks=checkpoint_callback,
    default_root_dir=root_dir,
)
# trainer = pytorch_lightning.Trainer(
#     gpus=[0],
#     limit_train_batches=1,
#     max_epochs=2000,
#     check_val_every_n_epoch=1,
#     callbacks=checkpoint_callback,
#     default_root_dir=root_dir,
# )
# ! >>>

# train
trainer.fit(net)

# ### Plot the loss and metric

eval_num = 250
plt.figure("train", (12, 6))
plt.subplot(1, 2, 1)
plt.title("Iteration Average Loss")
x = [eval_num * (i + 1) for i in range(len(net.epoch_loss_values))]
y = net.epoch_loss_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.subplot(1, 2, 2)
plt.title("Val Mean Dice")
x = [eval_num * (i + 1) for i in range(len(net.metric_values))]
y = net.metric_values
plt.xlabel("Iteration")
plt.plot(x, y)
plt.show()
plt.savefig('runs/fig.svg')

# ### Check best model output with the input image and label

# ! <<<
# slice_map = {
#     "img0035.nii.gz": 170,
#     "img0036.nii.gz": 230,
#     "img0037.nii.gz": 204,
#     "img0038.nii.gz": 204,
#     "img0039.nii.gz": 204,
#     "img0040.nii.gz": 180,
# }

# case_num = 4
# net.load_from_checkpoint(os.path.join(root_dir, "best_metric_model-v1.ckpt"))
# net.eval()
# net.to(device)

# with torch.no_grad():
#     img_name = os.path.split(
#         net.val_ds[case_num]["image_meta_dict"]["filename_or_obj"]
#     )[1]
#     img = net.val_ds[case_num]["image"]
#     label = net.val_ds[case_num]["label"]
#     val_inputs = torch.unsqueeze(img, 1).cuda(device)
#     val_labels = torch.unsqueeze(label, 1).cuda(device)
#     val_outputs = sliding_window_inference(
#         val_inputs, (96, 96, 96), 4, net, overlap=0.8
#     )
#     plt.figure("check", (18, 6))
#     plt.subplot(1, 3, 1)
#     plt.title(f"image")
#     plt.imshow(val_inputs.cpu().numpy()[0, 0, :, :, slice_map[img_name]], cmap="gray")
#     plt.subplot(1, 3, 2)
#     plt.title(f"label")
#     plt.imshow(val_labels.cpu().numpy()[0, 0, :, :, slice_map[img_name]])
#     plt.subplot(1, 3, 3)
#     plt.title(f"output")
#     plt.imshow(
#         torch.argmax(val_outputs, dim=1).detach().cpu()[0, :, :, slice_map[img_name]]
#     )
#     plt.show()

# ### Cleanup data directory
#
# Remove directory if a temporary was used.

# if directory is None:
#     shutil.rmtree(root_dir)
# ! >>>