from itertools import combinations
from pathlib import Path
from typing import Tuple, Optional, List, Dict, Any, Literal

import torch
from lightning import LightningModule, Trainer
from lightning.pytorch.callbacks import EarlyStopping, ModelSummary, ModelCheckpoint
from lightning.pytorch.utilities.types import OptimizerLRScheduler, STEP_OUTPUT

from loguru import logger

from torchmetrics.functional import accuracy, precision, recall, f1_score

from transformers import PreTrainedModel

from json import dump as json_dump

import PyModelUnits


def setup_train(module: LightningModule, root_path: Path,
                training_data, validation_data, test_data: Optional = None, metric_file_name: Optional[str] = None,
                monitoring_metric: Optional[str] = None, free_space:bool = False) -> None:
    """
    Set up the trainer and train the model -- and test it. CORE METHOD

    :param module: the model to train - the weights will be adapted, and the best model weights will be restored
    (according to the monitoring metric)
    :param root_path: the root path to save the model weights and stats
    :param training_data: the dataset including training data - see PyDataset.finish_datasets
    :param validation_data: the dataset including validation data - see PyDataset.finish_datasets
    :param test_data: the dataset including test data - see PyDataset.finish_datasets
    (if not given, the validation data is used for testing)
    (see here https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html /
    https://lightning.ai/docs/pytorch/stable/api/lightning.pytorch.core.LightningDataModule.html#lightning.pytorch.core.LightningDataModule.from_datasets)
    Here, you define the batch size, too.
    :param metric_file_name: the name of the file to store the metrics
    (if not given, the file is named "test_metrics_{TASK}"). Useful if you have other test data than the train data
    :param monitoring_metric: The metric which should be monitored to save the best model weights/ early stopping.
    If no metric is given, there is no early stopping (12 epochs), and the last model weights are saved
    :param free_space: If True, the saved model weights are deleted after the evaluation at the end of this method
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

    if not root_path.exists() or not root_path.joinpath("lightning_logs").exists():
        trainer.fit(module, train_dataloaders=training_data, val_dataloaders=validation_data)
        logger.success("Finished training: {} epochs, {} batches", trainer.current_epoch, trainer.global_step)
        logger.info("Last performance: {}",
                    "/".join([f"{k}: {round(v.cpu().item(), 3) if isinstance(v, torch.Tensor) else v}"
                              for k, v in trainer.logged_metrics.items()]))
    else:
        logger.warning("Root path {} already exists (trained), skipping training {}",
                       root_path.absolute(),
                       "and testing" if monitoring_metric is None else "and just load the best model weights")

    if monitoring_metric is not None:
        logger.info("Loading best model weights")
        model_weights_path = root_path.joinpath(f"model_weights-{monitoring_metric}")
        if model_weights_path.exists():
            saved_states = list(model_weights_path.glob("*.ckpt"))
            logger.info("Found {} saved states", len(saved_states))
            if len(saved_states) > 0:
                ckpt = torch.load(saved_states[-1])
                logger.success("Loading last saved state from \"{}\" (from epoch: {})",
                               saved_states[-1], ckpt.get("epoch", "unknown"))
                module.load_state_dict(ckpt["state_dict"])
            else:
                logger.debug("\"{}\" was not found...", model_weights_path.absolute())
                logger.error("No saved states found for the metric \"{}\", you might skip training because the path "
                             "already exists -- but with another used monitoring metric?", monitoring_metric)

    if test_data is not None:
        try:
            logger.info("Testing model with {} batches of test data", len(test_data))
        except RuntimeError:
            logger.opt(exception=True).trace("Testing model")
    else:
        try:
            logger.warning("No test data provided, using validation data for testing ({} batches)",
                           len(validation_data))
        except RuntimeError:
            logger.opt(exception=True).trace("No validation data provided, using training data for testing")
        test_data = validation_data
    listed_test_metrics = trainer.test(model=module, dataloaders=test_data, verbose=False, ckpt_path=None)
    logger.debug("Fetched {} metrics lists, containing {} metrics in total",
                 len(listed_test_metrics), sum(map(len, listed_test_metrics)))
    logger.success("Tested model, resulting into following metrics: {}",
                   "\nAND (other task)\n".join(map(lambda mkv: "\n".join(map(
                       lambda kv: f"\t{kv[0]}: {kv[1]:.3f}",
                       mkv.items())
                   ), listed_test_metrics)))

    logger.trace("Let's store the results in the root path")
    for i, metrics in enumerate(listed_test_metrics):
        with root_path.joinpath(
                f"test_metrics_{i}.txt" if metric_file_name is None else f"test_metrics_{metric_file_name}-{i}.txt"
        ).open(mode="w", encoding="utf-8") as fs:
            json_dump(obj=metrics, fp=fs, indent=2, sort_keys=True)

    if free_space:
        logger.debug("CLOSING === Freeing space by deleting model weights === CLOSING")
        model_weights_files = root_path.rglob(pattern="*.ckpt")
        for model_weights_file in model_weights_files:
            if model_weights_file.is_dir():
                logger.debug("Skipping directory {}", model_weights_file.name)
                continue
            logger.info(
                "We free space here: {} ({} MB)",
                model_weights_file,
                round(model_weights_file.stat().st_size/1024/1024, 1)
                if model_weights_file.stat().st_size >= 1024**2 else 0
            )
            if model_weights_file.stat().st_size >= 1024**2:
                model_weights_file.unlink()
        logger.debug("CLOSING === Freed space by deleting model weights === CLOSING")


class ClassificationModule(LightningModule):
    def __init__(self, core_model: torch.nn.Module, num_classes: int, task_name: str, learning_rate: float = 1e-3):
        """
        A classification module, which can be used to train a single-label classification task.
        :param core_model: the "heart" of the model, which should be a torch.nn.Module
        (e.g. a LLM, a RNN, a CNN, ...)
        :param num_classes: number of classes for the classification task (this classification module
        puts the classification head on top of the core model)
        :param task_name: the name of the tas to be processed - should be unique
        :param learning_rate: the learning rate for this classification module (optimizer)
        If you put this module into a MultiClassificationModule, the learning rate given here would be ignored
        """
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
            if isinstance(core_model, torch.nn.RNNBase):
                out_features = core_model.hidden_size*(1+int(core_model.bidirectional))
                logger.debug("Core model is RNN, using hidden size as input to classification layer ({}->{})",
                             out_features, num_classes)
            elif isinstance(core_model, PyModelUnits.LinearNNEncoder):
                out_features = core_model.in_layer.out_features*2
                logger.info("Core model is LinearNNEncoder, using the doubled output size (mean, std) as input to "
                            "classification layer ({}->{})",
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
            if torch.cuda.is_available() and "lengths" in kwargs:
                logger.trace("Lengths are moved on a GPU (mistaken from pytorch lightning), "
                             "moving back to CPU as it is required by Pytorch")
                kwargs["lengths"] = kwargs["lengths"].cpu()
            output_core = self.core_model(
                torch.nn.utils.rnn.pack_padded_sequence(**kwargs, batch_first=True, enforce_sorted=False)
            )[0]
            output_core = torch.stack(
                tensors=[instance[-1] for instance in torch.nn.utils.rnn.unpack_sequence(output_core)],
                dim=0
            )
        elif isinstance(self.core_model, PyModelUnits.LinearNNEncoder):
            output_core = self.core_model(**kwargs)
        else:
            logger.error("Core model is not supported yet: {}", self.core_model.__class__.__name__)
            output_core = torch.zeros((1, self.core_model.out_features))
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

    def metric_log(self, prediction: torch.Tensor, target: torch.Tensor, metric_logger: LightningModule,
                   split: Literal["train", "val", "test"] = "val"):
        metric_logger.log(
            name="{}_accMicro_{}".format(split, self.task_name),
            value=accuracy(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=1,
                           average="micro"),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log(
            name="{}_accTop3Micro_{}".format(split, self.task_name),
            value=accuracy(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=3,
                           average="micro"),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log(
            name="{}_precisionMacro_{}".format(split, self.task_name),
            value=precision(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=1,
                            average="macro"),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log(
            name="{}_recallMacro_{}".format(split, self.task_name),
            value=recall(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=1,
                         average="macro"),
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log(
            name="{}_f1Macro_{}".format(split, self.task_name),
            value=f1_score(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=1,
                           average="macro"),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log(
            name="{}_f1Micro_{}".format(split, self.task_name),
            value=f1_score(preds=prediction, target=target, task="multiclass", num_classes=self.num_classes, top_k=1,
                           average="micro"),
            prog_bar=True,
            logger=True,
            on_epoch=True,
            on_step=False
        )
        metric_logger.log_dict(
            dictionary={"{}_f1_{}=>Class{:0>2}".format(split, self.task_name, i): f1 for i, f1 in
                        enumerate(f1_score(
                            preds=prediction, target=target, task="multiclass",
                            num_classes=self.num_classes, top_k=1, average=None
                        ))},
            prog_bar=False,
            logger=True,
            on_epoch=True,
            on_step=False
        )

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
            self.metric_log(prediction=output, target=y, metric_logger=val_logger, split="val")
        return loss

    def test_step(self, batch, batch_idx):
        logger.trace("Testing step {} with batch {}", batch_idx, batch)
        y = batch.get("y", None)
        if y is None:
            logger.error("No labels for test step, only {}", "#".join(batch.keys()))
        val_logger = batch.pop("logger", self)
        logger.trace("Test-logging at {}", val_logger.logger.name)

        output, loss = self(batch)
        if y is not None:
            val_logger.log(name="test_loss_{}".format(self.task_name), value=loss, prog_bar=False, logger=False,
                           on_epoch=True)
            self.metric_log(prediction=output, target=y, metric_logger=val_logger, split="test")
            prediction_path = Path(".out").joinpath(".predictions").joinpath(self.task_name).joinpath("test set")
            prediction_path.mkdir(parents=True, exist_ok=True)
            prediction_path.joinpath("batch_{}.csv".format(batch_idx)).write_text(
                data="Predicted class,Predicted Prob,True class\n{}".format(
                    "\n".join([f"{torch.argmax(_o.cpu()).item()},"
                               f"{torch.max(_o.cpu()).item():.3f},"
                               f"{_y.cpu().item() if len(_y.shape) == 0 else _y.cpu().tolist()}"
                               for _o, _y in zip(output, y)])
                ),
                encoding="utf-8",
                errors="ignore"
            )
            logger.debug("Stored predictions in {} (batch: {})", prediction_path.absolute(), batch_idx)
        return loss


class MultiClassificationModule(LightningModule):
    def __init__(
            self,
            classification_modules: List[LightningModule],
            soft_parameter_sharing: Optional[float] = None,
            task_weighting: Optional[List[float]] = None,
            learning_rate: float = 1e-3
    ):
        """
        A multi-task classification module, which can be used to train multiple classification tasks at once.
        :param classification_modules: A list of classification modules (see PyModels.ClassificationModule).
        Each module should have a different name - be aware of needing a combined dataset for training this module!
        :param soft_parameter_sharing: Enable soft parameter sharing by setting a lambda value
        (normally in between 0.0-1.0)
        :param task_weighting: are different tasks more important than others? Or: if you have a task with fewer samples
        (cycling), you can consider this here
        :param learning_rate: the overall learning rate for this multi-task module
        """
        super().__init__()
        self.classification_modules = torch.nn.ModuleList(classification_modules)
        logger.info("Gathered {} classification modules", len(self.classification_modules))
        self.soft_parameter_sharing = soft_parameter_sharing
        if soft_parameter_sharing is not None:
            logger.debug("Using soft parameter sharing with gamma={}", soft_parameter_sharing)
        self.task_weighting = task_weighting
        if task_weighting is None:
            self.task_weighting = [1/len(classification_modules)] * len(classification_modules)
        else:
            logger.debug("Using task weighting with weights={}", task_weighting)
            if len(task_weighting) == 1:
                logger.info("Task weighting has only one value, using it for all classification modules -- "
                            "makes no sense!")
                self.task_weighting = None
            elif len(task_weighting) != len(classification_modules):
                logger.warning("Task weighting must have same length as classification modules: {} != {}",
                               len(task_weighting), len(classification_modules))
                if len(task_weighting) < len(classification_modules):
                    logger.info("Task weighting is shorter than classification modules, padding with 1s")
                    self.task_weighting.extend([1.0] * (len(classification_modules) - len(task_weighting)))
                elif len(task_weighting) > len(classification_modules):
                    logger.warning("Task weighting is longer than classification modules, truncating")
                    self.task_weighting = task_weighting[:len(classification_modules)]
            else:
                logger.debug("Task weighting has same length as classification modules ({}) -- good",
                             len(self.classification_modules))
            if self.task_weighting is not None and sum(self.task_weighting) != 1:
                logger.debug("Task weighting does not sum up to 1 but {}, normalizing", sum(self.task_weighting))
                self.task_weighting = [w/sum(self.task_weighting) for w in self.task_weighting]

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

    def test_step(self, batch, batch_idx):
        logger.trace("Testing step {} with batch {}", batch_idx, batch)
        outputs, loss = self(batch)
        self.log(name="test_lossAll", value=loss, prog_bar=False, logger=True, on_epoch=True)

        with torch.no_grad():
            for i, module in enumerate(self.classification_modules):
                if isinstance(module, LightningModule):
                    logger.debug("Testing metric score collection {} with batch {}", i, "#".join(batch[i].keys()))
                    batch[i]["logger"] = self
                    module.test_step(batch[i], batch_idx)
                    # logger.trace("Test-logged at {}", batch[i].pop("logger"))

        return loss
