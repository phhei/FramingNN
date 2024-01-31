from itertools import combinations
from pathlib import Path
from typing import Tuple, Optional, List, Union, Dict, Any

import torch
from lightning import LightningModule, Trainer, LightningDataModule
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from loguru import logger

from torchmetrics.functional import accuracy, precision, recall, f1_score

from transformers import PreTrainedModel


def setup_train(module: LightningModule, root_path: Path,
                dataset: LightningDataModule,
                monitoring_metric: Optional[str] = None) -> None:
    """
    Set up the trainer and train the model

    :param module: the model to train - the weights will be adapted, and the best model weights will be restored
    (according to the monitoring metric)
    :param root_path: the root path to save the model weights and stats
    :param dataset: the dataset including training and validation data
    (see here https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html /
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule.from_datasets)
    Here, you define the batch size, too.
    :param monitoring_metric: The metric which should be monitored to save the best model weights/ early stopping.
    If no metric is given, there is no early stopping (12 epochs), and the last model weights are saved
    :return: Nothing - the model weights are adapted in place
    """

    callbacks = [
        ModelSummary(max_depth=2),
    ]
    if monitoring_metric is not None:
        callbacks.append(
            ModelCheckpoint(
                monitor=monitoring_metric,
                dirpath=root_path.joinpath("model_weights"),
                filename="best_model.pt",
                save_top_k=1,
                mode="min" if "loss" in monitoring_metric else "max",
                save_weights_only=True,
                verbose=True
            )
        )
        callbacks.append(
            EarlyStopping(
                monitor=monitoring_metric,
                patience=2,
                verbose=True,
                mode="min" if "loss" in monitoring_metric else "max",
                check_on_train_epoch_end=False
            )
        )
    else:
        callbacks.append(
            ModelCheckpoint(
                dirpath=root_path.joinpath("model_weights"),
                verbose=False,
                save_last=True,
                save_weights_only=True
            ))

    logger.info("Setting up trainer with {} callbacks: {}", len(callbacks), callbacks)

    trainer = Trainer(
        default_root_dir=root_path,
        min_epochs=2,
        max_epochs=12,
        callbacks=callbacks,
    )

    logger.success("Setting up trainer: {}", trainer)

    trainer.fit(module, datamodule=dataset)
    print(trainer.logged_metrics)

    logger.success("Finished training: {} epochs, {} batches", trainer.current_epoch, trainer.global_step)
    logger.info("Last performance: {}",
                "/".join([f"{k}: {round(v.cpu().item(), 3) if isinstance(v, torch.Tensor) else v}"
                          for k, v in trainer.logged_metrics.items()]))
    if monitoring_metric is not None:
        logger.info("Loading best model weights")
        module.load_state_dict(torch.load(root_path.joinpath("model_weights", "best_model.pt")))


class ClassificationModule(LightningModule):
    def __init__(self, core_model: torch.nn.Module, num_classes: int, task_name: str, learning_rate: float = 1e-3):
        super().__init__()

        logger.info("Initializing classification module for task {} with {} classes", task_name, num_classes)
        self.task_name = task_name

        self.core_model = core_model
        if isinstance(self.core_model, PreTrainedModel):
            logger.info("Core model is a pretrained model \"{}\"", self.core_model.config.architectures[0])
            logger.trace("Not AutoModelForSequenceClassification.from_pretrained("
                         "XXX, num_labels=<num_classes>, return_dict=False, "
                         "problem_type=\"single_label_classification\") but AutoModel.from_pretrained("
                         "XXX, return_dict=True)")
            out_features = self.core_model.config.hidden_size
        else:
            if isinstance(core_model, torch.nn.LSTM) or isinstance(core_model, torch.nn.GRU):
                out_features = core_model.hidden_size*(1+int(core_model.bidirectional))
                logger.debug("Core model is RNN, using hidden size as input to classification layer ({}->{})",
                             out_features, num_classes)
            elif isinstance(core_model, torch.nn.Linear):
                out_features = core_model.out_features
                logger.info("Core model is Linear (unusual), using output size as input "
                            "to classification layer ({}->{})",
                            out_features, num_classes)
            else:
                raise ValueError("Core model must be a Linear or RNN layer")
        self.num_classes = num_classes
        self.dropout = torch.nn.Dropout(p=0.2)
        self.fc = torch.nn.Linear(in_features=out_features, out_features=num_classes, bias=True)
        self.normalizer = torch.nn.Softmax(dim=1)

        self.loss = torch.nn.CrossEntropyLoss()

        self.learning_rate = learning_rate

    def forward(self, x: Union[torch.Tensor, Dict[str, Any]], y: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if isinstance(self.core_model, PreTrainedModel):
            output_logits = self.fc(self.dropout(self.core_model(**x).last_hidden_state[:, 0, :]))
        elif isinstance(self.core_model, torch.nn.RNNBase):
            output_rnn = self.core_model(
                torch.nn.utils.rnn.pack_padded_sequence(
                    x[:, 1:, :], lengths=x[:, 0, 0].cpu(), batch_first=True, enforce_sorted=False
                )
            )[0]
            output_rnn = torch.stack(
                tensors=[instance[-1] for instance in torch.nn.utils.rnn.unpack_sequence(output_rnn)],
                dim=0
            )
            output_logits = self.fc(self.dropout(output_rnn))
        else:
            output_rnn = self.core_model(x)[0]
            assert isinstance(output_rnn, torch.Tensor)
            output_rnn = output_rnn[:, -1, :]
            output_logits = self.fc(self.dropout(output_rnn))
        if y is None:
            logger.trace("No labels provided, returning only output, no loss")
        return self.normalizer(output_logits), None if y is None else self.loss(output_logits, y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25, patience=1, verbose=True)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_loss_{}".format(self.task_name),
                "strict": False,
                "name": "LRObserver"
            }
        }

    def training_step(self, batch, batch_idx):
        logger.trace("Training step {} with batch {}", batch_idx, batch)
        x, y = batch
        output, loss = self(x, y)
        self.log(
            name="train_loss_{}".format(self.task_name),
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logger.trace("Validation step {} with batch {}", batch_idx, batch)
        x, y = batch
        output, loss = self(x, y)
        self.log(name="val_loss_{}".format(self.task_name), value=loss, prog_bar=False, logger=True, on_epoch=True)
        self.log(
            name="val_accMacro_{}".format(self.task_name),
            value=accuracy(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
            prog_bar=False,
            logger=True,
            on_epoch=True
        )
        self.log(
            name="val_precisionMacro_{}".format(self.task_name),
            value=precision(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
            prog_bar=False,
            logger=True,
            on_epoch=True
        )
        self.log(
            name="val_recallMacro_{}".format(self.task_name),
            value=recall(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
            prog_bar=False,
            logger=True,
            on_epoch=True
        )
        self.log(
            name="val_f1Macro_{}".format(self.task_name),
            value=f1_score(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
            prog_bar=True,
            logger=True,
            on_epoch=True
        )
        self.log(
            name="val_f1Micro_{}".format(self.task_name),
            value=f1_score(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="micro"),
            prog_bar=False,
            logger=True,
            on_epoch=True
        )
        return loss


class DoubleClassificationModule(LightningModule):
    def __init__(
            self,
            classification_modules: List[LightningModule],
            soft_parameter_sharing: Optional[float],
            task_weighting: Optional[List[float]] = None,
            learning_rate: float = 1e-3
    ):
        super().__init__()
        self.classification_modules = torch.nn.ModuleList(classification_modules)
        logger.info("Gathered {} classification modules", len(self.classification_modules))
        self.soft_parameter_sharing = soft_parameter_sharing
        if soft_parameter_sharing is not None:
            logger.debug("Using soft parameter sharing with gamma={}", soft_parameter_sharing)
        self.task_weighting = task_weighting
        if self.task_weighting is not None:
            logger.debug("Using task weighting with weights={}", task_weighting)
            if len(task_weighting) != len(classification_modules):
                logger.warning("Task weighting must have same length as classification modules: {} != {}",
                               len(task_weighting), len(classification_modules))
                if len(task_weighting) < len(classification_modules):
                    logger.info("Task weighting is shorter than classification modules, padding with 1s")
                    self.task_weighting.extend([1.0] * (len(classification_modules) - len(task_weighting)))
                elif len(task_weighting) > len(classification_modules):
                    logger.warning("Task weighting is longer than classification modules, truncating")
                    self.task_weighting = task_weighting[:len(classification_modules)]

        self.learning_rate = learning_rate

    def forward(self, x: List[torch.Tensor], y: List[Optional[torch.Tensor]]) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        assert len(x) == len(self.classification_modules), (
            "Input must be a list of {} tensors".format(len(self.classification_modules)))
        outputs = []
        losses = []
        for _i, _x in enumerate(x):
            assert isinstance(_x, torch.Tensor), "Input must be a list of tensors"
            _output, _loss = self.classification_modules[_i](_x, None if y is None else y[_i])
            logger.debug("Module {} output: {} (loss: {})", _i, _output.shape, _loss)
            outputs.append(_output)
            if _loss is not None:
                losses.append(_loss if self.task_weighting is None else (_loss * self.task_weighting[_i]))

        loss = torch.scalar_tensor(s=0, dtype=torch.float) if len(losses) == 0 else torch.stack(losses).mean()
        if self.soft_parameter_sharing is not None:
            logger.trace("Using soft parameter sharing, increase {}", loss)

            for m1, m2 in combinations(self.classification_modules, 2):
                assert isinstance(m1, torch.nn.Module) and isinstance(m2, torch.nn.Module)
                for p1, p2 in zip(m1.parameters(recurse=True), m2.parameters(recurse=True)):
                    assert isinstance(p1, torch.Tensor) and isinstance(p2, torch.Tensor)
                    loss_add = self.soft_parameter_sharing * torch.sum(torch.pow(p1 - p2, 2))
                    loss = loss + loss_add
                    logger.trace("Parameter {} has a difference loss of {}", p1.shape, loss_add)

        return outputs, None if all(map(lambda _y: _y is None, y)) and self.soft_parameter_sharing is None else loss

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.2, patience=1, verbose=True
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
                "monitor": "val_lossAll",
                "strict": False,
                "name": "LRObserverMultiTask"
            }
        }

    def training_step(self, batch, batch_idx):
        logger.trace("Training step {} with batch {}", batch_idx, batch)
        x, y = batch
        outputs, loss = self(x, y)
        self.log(
            name="train_lossAll",
            value=loss,
            prog_bar=True,
            logger=True,
            on_step=True,
            on_epoch=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        logger.trace("Validation step {} with batch {}", batch_idx, batch)
        x, y = batch
        outputs, loss = self(x, y)
        self.log(name="val_lossAll", value=loss, prog_bar=False, logger=True, on_epoch=True)

        with torch.no_grad():
            for module in self.classification_modules:
                if isinstance(module, LightningModule):
                    module.validation_step(batch, batch_idx)

        return loss
