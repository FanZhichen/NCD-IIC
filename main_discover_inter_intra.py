import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from pytorch_lightning.metrics import Accuracy

from utils.data import get_datamodule
from utils.nets import MultiHeadResNet
from utils.eval import ClusterMetrics
from utils.sinkhorn_knopp import SinkhornKnopp

import numpy as np
from argparse import ArgumentParser
from datetime import datetime
from scipy.special import comb

parser = ArgumentParser()
# file path
parser.add_argument("--entity", default='fanzhichen', type=str, help="wandb entity")
parser.add_argument("--project", default="iic", type=str, help="wandb project")
parser.add_argument("--data_dir", default="/data/fzc", type=str, help="dataset directory")
parser.add_argument("--log_dir", default="logs", type=str, help="log directory")
parser.add_argument("--pretrained", type=str, help="pretrained checkpoint path")  # Use pretrained checkpoint
# dataset splits
parser.add_argument("--dataset", default="CIFAR10", type=str, help="dataset")
parser.add_argument("--factor_inter", default=0.05, type=float, help="factor for inter-class sKLD")
parser.add_argument("--factor_intra", default=0.01, type=float, help="factor for intra-class sKLD")
parser.add_argument("--num_labeled_classes", default=80, type=int, help="number of labeled classes")
parser.add_argument("--num_unlabeled_classes", default=20, type=int, help="number of unlab classes")
parser.add_argument("--imagenet_split", default="A", type=str, help="imagenet split [A,B,C]")  # Only for ImageNet
# hyperparameters
parser.add_argument("--download", default=False, action="store_true", help="whether to download")
parser.add_argument("--offline", default=False, action="store_true", help="disable wandb")  # True when debugging
parser.add_argument("--num_workers", default=5, type=int, help="number of workers")
parser.add_argument("--warmup_epochs", default=10, type=int, help="warmup epochs")  # 5 for ImageNet, 10 for the others
parser.add_argument("--batch_size", default=256, type=int, help="batch size")  # 256 for ImageNet, 512 for the others
parser.add_argument("--arch", default="resnet18", type=str, help="backbone architecture")
parser.add_argument("--base_lr", default=0.4, type=float, help="learning rate")  # 0.2 for ImageNet, 0.4 for the others
parser.add_argument("--min_lr", default=0.001, type=float, help="min learning rate")
parser.add_argument("--momentum_opt", default=0.9, type=float, help="momentum for optimizer")
parser.add_argument("--weight_decay_opt", default=1.5e-4, type=float, help="weight decay")
parser.add_argument("--temperature", default=0.1, type=float, help="softmax temperature")
parser.add_argument("--comment", default=datetime.now().strftime("%b%d_%H-%M-%S"), type=str)
parser.add_argument("--proj_dim", default=256, type=int, help="projected dim")  # MLP output dim
parser.add_argument("--hidden_dim", default=2048, type=int, help="hidden dim in proj/pred head")  # MLP hidden dim
parser.add_argument("--num_hidden_layers", default=1, type=int, help="number of hidden layers")  # MLP hidden layers
parser.add_argument("--overcluster_factor", default=3, type=int, help="overclustering factor")  # Overclustering
parser.add_argument("--num_heads", default=4, type=int, help="number of heads for clustering")  # Multi-head clustering
parser.add_argument("--num_iters_sk", default=3, type=int, help="number of iters for Sinkhorn")  # For SK algorithm
parser.add_argument("--epsilon_sk", default=0.05, type=float, help="epsilon for the Sinkhorn")  # For SK algorithm
parser.add_argument("--multicrop", default=True, action="store_true", help="activates multicrop")  # New for UNOv2
parser.add_argument("--num_large_crops", default=2, type=int, help="number of large crops")
parser.add_argument("--num_small_crops", default=2, type=int, help="number of small crops")


class Discoverer(pl.LightningModule):
    def __init__(self, **kwargs):
        super().__init__()
        self.save_hyperparameters({k: v for (k, v) in kwargs.items() if not callable(v)})

        # build model
        self.model = MultiHeadResNet(
            arch=self.hparams.arch,
            low_res="CIFAR" in self.hparams.dataset,
            num_labeled=self.hparams.num_labeled_classes,
            num_unlabeled=self.hparams.num_unlabeled_classes,
            proj_dim=self.hparams.proj_dim,
            hidden_dim=self.hparams.hidden_dim,
            overcluster_factor=self.hparams.overcluster_factor,
            num_heads=self.hparams.num_heads,
            num_hidden_layers=self.hparams.num_hidden_layers,
        )

        state_dict = torch.load(self.hparams.pretrained, map_location=self.device)
        state_dict = {k: v for k, v in state_dict.items() if ("unlab" not in k)}
        self.model.load_state_dict(state_dict, strict=False)

        # Sinkorn-Knopp
        self.sk = SinkhornKnopp(
            num_iters=self.hparams.num_iters_sk, epsilon=self.hparams.epsilon_sk
        )

        # metrics
        self.metrics = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )
        self.metrics_inc = torch.nn.ModuleList(
            [
                ClusterMetrics(self.hparams.num_heads),
                ClusterMetrics(self.hparams.num_heads),
                Accuracy(),
            ]
        )

        # buffer for best head tracking
        self.register_buffer("loss_per_head", torch.zeros(self.hparams.num_heads))

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

    def cross_entropy_loss(self, preds, targets):
        preds = F.log_softmax(preds / self.hparams.temperature, dim=-1)
        # return -torch.mean(torch.sum(targets * preds, dim=-1))
        return torch.mean(-torch.sum(targets * preds, dim=-1), dim=-1)

    def swapped_prediction(self, logits, targets):
        loss = 0
        for view in range(self.hparams.num_large_crops):
            for other_view in np.delete(range(self.hparams.num_crops), view):
                loss += self.cross_entropy_loss(logits[other_view], targets[view])
        return loss / (self.hparams.num_large_crops * (self.hparams.num_crops - 1))

    def intra_class_sKLD(self, logits, mask_lab):
        log_prob_lab = F.log_softmax(logits[:, :, mask_lab, :self.hparams.num_labeled_classes] / self.hparams.temperature, dim=-1)
        log_prob_unlab = F.log_softmax(logits[:, :, ~mask_lab, self.hparams.num_labeled_classes:] / self.hparams.temperature, dim=-1)

        kl_loss = 0
        for view_i in range(self.hparams.num_crops):
            log_prob_lab_view_i = log_prob_lab[view_i]
            log_prob_unlab_view_i = log_prob_unlab[view_i]
            for view_j in range(view_i + 1, self.hparams.num_crops):
                log_prob_lab_view_j = log_prob_lab[view_j]
                log_prob_unlab_view_j = log_prob_unlab[view_j]
                # reduction='batchmean': divide the calculated result by num_heads=4,
                # then the result will be divided by the number of labelled samples and labelled classes
                kl_loss_lab = (F.kl_div(log_prob_lab_view_i, torch.exp(log_prob_lab_view_j), reduction='batchmean')
                               + F.kl_div(log_prob_lab_view_j, torch.exp(log_prob_lab_view_i), reduction='batchmean')
                               ) / (2 * mask_lab.sum().item() * self.hparams.num_labeled_classes)
                # then the result will be divided by the number of unlabelled samples and unlabelled classes
                kl_loss_unlab = (F.kl_div(log_prob_unlab_view_i, torch.exp(log_prob_unlab_view_j), reduction='batchmean')
                                 + F.kl_div(log_prob_unlab_view_j, torch.exp(log_prob_unlab_view_i), reduction='batchmean')
                                 ) / (2 * (~mask_lab).sum().item() * self.hparams.num_unlabeled_classes)
                kl_loss += (kl_loss_lab + kl_loss_unlab)
        return kl_loss / comb(self.hparams.num_crops, 2, exact=True)

    def inter_class_sKLD(self, logits, mask_lab):
        log_prob_lab = F.log_softmax(logits[:, :, mask_lab, :] / self.hparams.temperature, dim=-1)
        log_prob_unlab = F.log_softmax(logits[:, :, ~mask_lab, :] / self.hparams.temperature, dim=-1)

        kl_loss = 0
        for view in range(self.hparams.num_crops):
            kl_view = 0
            for head in range(self.hparams.num_heads):
                log_prob_lab_vh = log_prob_lab[view][head]
                log_prob_unlab_vh = log_prob_unlab[view][head]
                kl_view += (
                                   torch.mean(
                                       torch.diag(torch.mm(torch.exp(log_prob_lab_vh), log_prob_lab_vh.t())).view(-1, 1) -
                                       torch.mm(torch.exp(log_prob_lab_vh), log_prob_unlab_vh.t())
                                   ) +
                                   torch.mean(
                                       torch.diag(torch.mm(torch.exp(log_prob_unlab_vh), log_prob_unlab_vh.t())).view(-1, 1) -
                                       torch.mm(torch.exp(log_prob_unlab_vh), log_prob_lab_vh.t())
                                   )
                           ) / 2
            kl_loss += kl_view / self.hparams.num_heads
        return -(kl_loss / self.hparams.num_crops)

    def forward(self, x):
        return self.model(x)

    def on_epoch_start(self):
        self.loss_per_head = torch.zeros_like(self.loss_per_head)

    def unpack_batch(self, batch):
        if self.hparams.dataset == "ImageNet":
            views_lab, labels_lab, views_unlab, labels_unlab = batch
            views = [torch.cat([vl, vu]) for vl, vu in zip(views_lab, views_unlab)]
            labels = torch.cat([labels_lab, labels_unlab])
        else:
            views, labels = batch
        mask_lab = labels < self.hparams.num_labeled_classes
        return views, labels, mask_lab

    def training_step(self, batch, _):
        # views: (2*num_large_crops=4, batch_size, C, H, W)
        views, labels, mask_lab = self.unpack_batch(batch)
        nlc = self.hparams.num_labeled_classes

        # normalize prototypes
        self.model.normalize_prototypes()

        # forward
        outputs = self.model(views)

        # gather outputs
        outputs["logits_lab"] = (outputs["logits_lab"].unsqueeze(1).expand(-1, self.hparams.num_heads, -1, -1))
        # concatenate logtis: (2*num_large_crops, num_heads, batch_size, num_labeled_classes + num_unlabeled_classes)
        logits = torch.cat([outputs["logits_lab"], outputs["logits_unlab"]], dim=-1)
        # logtis_over: (2*num_large_crops, num_heads, batch_size, num_labeled_classes + num_unlabeled_classes * overcluster_factor)
        logits_over = torch.cat([outputs["logits_lab"], outputs["logits_unlab_over"]], dim=-1)

        # create targets
        targets_lab = (
            F.one_hot(labels[mask_lab], num_classes=self.hparams.num_labeled_classes)
                .float()
                .to(self.device)
        )
        targets = torch.zeros_like(logits)
        targets_over = torch.zeros_like(logits_over)

        # generate pseudo-labels with sinkhorn-knopp and fill unlabeled targets
        for v in range(self.hparams.num_large_crops):
            for h in range(self.hparams.num_heads):
                targets[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets_over[v, h, mask_lab, :nlc] = targets_lab.type_as(targets)
                targets[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab"][v, h, ~mask_lab]
                ).type_as(targets)
                targets_over[v, h, ~mask_lab, nlc:] = self.sk(
                    outputs["logits_unlab_over"][v, h, ~mask_lab]
                ).type_as(targets)

        # compute swapped prediction loss
        loss_cluster = self.swapped_prediction(logits, targets)
        loss_overcluster = self.swapped_prediction(logits_over, targets_over)

        # calculate inter-class and intra-class sKLD constraints
        inter_loss = self.inter_class_sKLD(logits, mask_lab)
        intra_loss = self.intra_class_sKLD(logits, mask_lab)

        # update best head tracker
        self.loss_per_head += loss_cluster.clone().detach()

        # total loss
        loss_cluster = loss_cluster.mean()
        loss_overcluster = loss_overcluster.mean()
        loss = (loss_cluster + loss_overcluster) / 2 + inter_loss * self.hparams.factor_inter + intra_loss * self.hparams.factor_intra

        # log
        results = {
            "loss": loss.detach(),
            "loss_cluster": loss_cluster.mean(),
            "loss_overcluster": loss_overcluster.mean(),
            "lr": self.trainer.optimizers[0].param_groups[0]["lr"],
            "inter_loss": inter_loss.mean(),
            "intra_loss": intra_loss.mean(),
        }

        self.log_dict(results, on_step=False, on_epoch=True, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx, dl_idx):
        images, labels = batch
        tag = self.trainer.datamodule.dataloader_mapping[dl_idx]

        # forward
        outputs = self(images)

        if "unlab" in tag:  # use clustering head
            preds = outputs["logits_unlab"]
            preds_inc = torch.cat(
                [
                    outputs["logits_lab"].unsqueeze(0).expand(self.hparams.num_heads, -1, -1),
                    outputs["logits_unlab"],
                ],
                dim=-1,
            )
        else:  # use supervised classifier
            preds = outputs["logits_lab"]
            best_head = torch.argmin(self.loss_per_head)
            preds_inc = torch.cat(
                [outputs["logits_lab"], outputs["logits_unlab"][best_head]], dim=-1
            )
        preds = preds.max(dim=-1)[1]
        preds_inc = preds_inc.max(dim=-1)[1]

        self.metrics[dl_idx].update(preds, labels)
        self.metrics_inc[dl_idx].update(preds_inc, labels)

    def validation_epoch_end(self, _):
        results = [m.compute() for m in self.metrics]
        results_inc = [m.compute() for m in self.metrics_inc]
        # log metrics
        for dl_idx, (result, result_inc) in enumerate(zip(results, results_inc)):
            prefix = self.trainer.datamodule.dataloader_mapping[dl_idx]
            prefix_inc = "incremental/" + prefix
            if "unlab" in prefix:
                for (metric, values), (_, values_inc) in zip(result.items(), result_inc.items()):
                    name = "/".join([prefix, metric])
                    name_inc = "/".join([prefix_inc, metric])
                    avg = torch.stack(values).mean()
                    avg_inc = torch.stack(values_inc).mean()
                    best = values[torch.argmin(self.loss_per_head)]
                    best_inc = values_inc[torch.argmin(self.loss_per_head)]
                    self.log(name + "/avg", avg, sync_dist=True)
                    self.log(name + "/best", best, sync_dist=True)
                    self.log(name_inc + "/avg", avg_inc, sync_dist=True)
                    self.log(name_inc + "/best", best_inc, sync_dist=True)
            else:
                self.log(prefix + "/acc", result)
                self.log(prefix_inc + "/acc", result_inc)


def main(args):
    dm = get_datamodule(args, "discover")

    run_name = "-".join(["discover", args.arch, args.dataset, args.comment])
    wandb_logger = pl.loggers.WandbLogger(
        save_dir=args.log_dir,
        name=run_name,
        project=args.project,
        entity=args.entity,
        offline=args.offline,
    )

    model = Discoverer(**args.__dict__)
    trainer = pl.Trainer.from_argparse_args(args, logger=wandb_logger)
    trainer.fit(model, dm)


if __name__ == "__main__":
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()
    args.num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if not args.multicrop:
        args.num_small_crops = 0
    args.num_crops = args.num_large_crops + args.num_small_crops

    main(args)