from itertools import combinations
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler

from loguru import logger

from torchmetrics.functional import accuracy, precision, recall, f1_score

from transformers import PreTrainedModel


def setup_train(module: LightningModule, root_path: Path,
                training_data, validation_data,
                monitoring_metric: Optional[str] = None) -> None:
    """
    Set up the trainer and train the model

    :param module: the model to train - the weights will be adapted, and the best model weights will be restored
    (according to the monitoring metric)
    :param root_path: the root path to save the model weights and stats
    :param training_data: the dataset including training data - see PyDataset.finish_datasets
    :param validation_data: the dataset including validation data - see PyDataset.finish_datasets
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
                dirpath=root_path.joinpath(f"model_weights-{monitoring_metric}"),
                filename=str(f"score[{monitoring_metric}]").replace("[", "{").replace("]", "}"),
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
                filename="epoch{epoch}.{step}_valLoss{val_loss:.2f}",
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

    trainer.fit(module, train_dataloaders=training_data, val_dataloaders=validation_data)
    print(trainer.logged_metrics)

    logger.success("Finished training: {} epochs, {} batches", trainer.current_epoch, trainer.global_step)
    logger.info("Last performance: {}",
                "/".join([f"{k}: {round(v.cpu().item(), 3) if isinstance(v, torch.Tensor) else v}"
                          for k, v in trainer.logged_metrics.items()]))
    if monitoring_metric is not None:
        logger.info("Loading best model weights")
        saved_states = list(root_path.joinpath(f"model_weights-{monitoring_metric}").glob("*.ckpt"))
        logger.info("Found {} saved states", len(saved_states))
        if len(saved_states) > 0:
            ckpt = torch.load(saved_states[-1])
            logger.success("Loading last saved state from \"{}\" (from epoch: {})",
                           saved_states[-1], ckpt.get("epoch", "unknown"))
            module.load_state_dict(ckpt["state_dict"])


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

    def forward(self, kwargs) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        def _pop_tensor_from_kwargs(key: List[str]) -> Optional[torch.Tensor]:
            logger.trace("Extracting tensor from kwargs with [{}]", "|".join(kwargs.keys()))

            for k in key:
                if k in kwargs:
                    logger.trace("Found key \"{}\" in kwargs", k)
                    return kwargs.pop(k)
                else:
                    logger.info("Key \"{}\" contained", k)

            logger.warning("{} not contained, no resulting vector", "->".join(key))
            return None

        y = _pop_tensor_from_kwargs(key=["y", "labels", "label_ids"])

        if isinstance(self.core_model, PreTrainedModel):
            output_core = self.core_model(**kwargs).last_hidden_state[:, 0, :]
        elif isinstance(self.core_model, torch.nn.RNNBase):
            output_core = self.core_model(
                torch.nn.utils.rnn.pack_padded_sequence(**kwargs, batch_first=True, enforce_sorted=False)
            )[0]
            output_core = torch.stack(
                tensors=[instance[-1] for instance in torch.nn.utils.rnn.unpack_sequence(output_core)],
                dim=0
            )
        else:
            x = _pop_tensor_from_kwargs(key=["x", "input"])
            if x is None:
                logger.error("No input provided, expect \"x\" or \"input\"")
                output_core = torch.zeros((1, self.core_model.hidden_size))
            else:
                output_core = self.core_model(x)[0]
                output_core = output_core[:, -1, :]
        output_logits = self.fc(self.dropout(output_core))
        if y is None:
            logger.trace("No labels provided, returning only output, no loss")
        return self.normalizer(output_logits), None if y is None else self.loss(output_logits, y)

    def configure_optimizers(self) -> OptimizerLRScheduler:
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.25,
                                                               patience=1, verbose=True)
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
        # if isinstance(batch, List):
        #     batch = batch[0]
        output, loss = self(batch)
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
        # if isinstance(batch, List):
        #    batch = batch[0]
        y = batch.get("y", None)
        if y is None:
            logger.warning("No labels for validation step, only {}", "#".join(batch.keys()))
        val_logger = batch.pop("logger", self)
        logger.trace("Val-logging at {}", val_logger.logger.name)

        output, loss = self(batch)
        if y is not None:
            val_logger.log(name="val_loss_{}".format(self.task_name), value=loss, prog_bar=False, logger=True, on_epoch=True)
            val_logger.log(
                name="val_accMacro_{}".format(self.task_name),
                value=accuracy(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False
            )
            val_logger.log(
                name="val_precisionMacro_{}".format(self.task_name),
                value=precision(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=False
            )
            val_logger.log(
                name="val_recallMacro_{}".format(self.task_name),
                value=recall(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
                prog_bar=False,
                logger=True,
                on_epoch=True,
                on_step=False
            )
            val_logger.log(
                name="val_f1Macro_{}".format(self.task_name),
                value=f1_score(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="macro"),
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False
            )
            val_logger.log(
                name="val_f1Micro_{}".format(self.task_name),
                value=f1_score(preds=output, target=y, task="multiclass", num_classes=self.num_classes, average="micro"),
                prog_bar=True,
                logger=True,
                on_epoch=True,
                on_step=False
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
            else:
                logger.warning("Task weighting has same length as classification modules ({}) -- deactivating",
                               len(self.classification_modules))
                self.task_weighting = None

        self.learning_rate = learning_rate

    def forward(self, listed_kwargs: Dict) -> Tuple[List[torch.Tensor], Optional[torch.Tensor]]:
        assert len(listed_kwargs) == len(self.classification_modules), (
            "Input must be a task-list of {} tensor-dicts".format(len(self.classification_modules)))
        outputs = []
        losses = []
        for _i, _kwargs in enumerate(listed_kwargs):
            assert isinstance(_kwargs, Dict), "Input must be a dict of tensors"

            y = _kwargs.get("y", None)
            _output, _loss = self.classification_modules[_i](_kwargs)
            if y is not None:
                _kwargs["y"] = y
            if _loss is not None:
                self.log(name="train_loss_m{}".format(_i), value=_loss, prog_bar=False, logger=True, on_epoch=True)

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

        return outputs, None if torch.sum(torch.abs(loss)) == 0 and self.soft_parameter_sharing is None else loss

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
        outputs, loss = self(batch)
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
        outputs, loss = self(batch)
        self.log(name="val_lossAll", value=loss, prog_bar=False, logger=True, on_epoch=True)

        with torch.no_grad():
            for i, module in enumerate(self.classification_modules):
                if isinstance(module, LightningModule):
                    logger.debug("Validating metric score collection {} with batch {}", i, "#".join(batch[i].keys()))
                    batch[i]["logger"] = self
                    module.validation_step(batch[i], batch_idx)
                    # logger.trace("Val-logged at {}", batch[i].pop("logger"))

        return loss
