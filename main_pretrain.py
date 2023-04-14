import torch
import torch.nn.functional as F
import torchvision
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.callbacks import PretrainCheckpointCallback

from argparse import ArgumentParser
from datetime import datetime


parser = ArgumentParser()
# file path
parser.add_argument("--entity", default="fanzhichen", type=str, help="wandb entity")
parser.add_argument("--project", default="iic", type=str, help="wandb project")
parser.add_argument("--data_dir", default="/data/fzc", type=str, help="dataset directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--checkpoint_dir", default="checkpoints_iic", type=str, help="checkpoint directory")  # To save pretrained checkpoint
# dataset splits
parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")  # CIFAR10, CIFAR100, ImageNet
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlabeled classes")
# hyperparameters
parser.add_argument("--download", default=False, action="store_true", help="whether to download dataset")
parser.add_argument("--pretrained", default=None, type=str, help="pretrained checkpoint path")  # Not used in pretrain
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")
parser.add_argument("--num_workers", default=5, type=int, help="number of workers")  # 8 for ImageNet, 5 for the others
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")  # 5 for ImageNet, 10 for the others
parser.add_argument("--batch_size", default=256, type=int, help="batch size")
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.1, type=float, help="learning rate")
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.0e-4, type=float, help="weight decay")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)


class Pretrainer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            num_heads=None,
        )

        if self.hparams.pretrained is not None:
            state_dict = torch.load(self.hparams.pretrained)
            self.model.load_state_dict(state_dict, strict=False)

        # metrics
        self.accuracy = Accuracy()

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=self.hparams.base_lr,
            momentum=self.hparams.momentum_opt,
            weight_decay=self.hparams.weight_decay_opt,
        )
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=self.hparams.warmup_epochs,
            max_epochs=self.hparams.max_epochs,
            warmup_start_lr=self.hparams.min_lr,
            eta_min=self.hparams.min_lr,
        )
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        images, labels = batch

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(images)

        # supervised loss
        loss_supervised = torch.stack(
            [F.cross_entropy(o / self.hparams.temperature, labels) for o in outputs["logits_lab"]]
        ).mean()

        # log
        results = {
            "loss_supervised": loss_supervised,
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
        }
        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)

        # reweight loss
        return loss_supervised

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        # forward
        logits = self.model(images)["logits_lab"]
        _, preds = logits.max(dim=-1)

        # calculate loss and accuracy
        loss_supervised = F.cross_entropy(logits, labels)
        acc = self.accuracy(preds, labels)

        # log
        results = {
            "val/loss_supervised": loss_supervised,
            "val/acc": acc,
        }
        self.log_dict(results, on_step=False, on_epoch=True)
        return results


def main(args):
    dm = get_datamodule(args, mode="pretrain")

    # logger
    run_name = "-".join(["pretrain", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Pretrainer(**args.__dict__)
    # initialize trainer
    trainer = pl.Trainer.from_argparse_args(
        args, logger=wandb_logger, callbacks=[PretrainCheckpointCallback()]
    )
    trainer.fit(model, datamodule=dm)


if __name__ == "__main__":
    # add available command parameters of Trainer to argparse
    # such as gpus, precision, max_epochs, distributed_backend, sync_batchnorm, etc
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    main(args)
