# System
import json
import logging
import os
import random
import sys

# Config
import hydra
import lightning as L
import numpy as np
import pandas as pd

# External
import torch
import torchvision

# DETOXAI
from detoxai import debias, download_datasets
from detoxai.core.model_wrappers import FairnessLightningWrapper
from detoxai.core.results_class import CorrectionResult
from detoxai.core.xai import RRF
from detoxai.datasets.catalog.download import SUPPORTED_DATASETS
from detoxai.metrics.fairness_metrics import (
    DEFAULT_METRICS_CONFIG,
    AllMetrics,
)
from detoxai.utils.dataloader import DetoxaiDataLoader
from detoxai.utils.datasets import (
    DetoxaiDataset,
    get_detoxai_datasets,
    make_detoxai_datasets_variant,
)
from detoxai.utils.transformations import SquarePad
from detoxai.visualization import (
    ConditionOn,
    DataVisualizer,
    HeatmapVisualizer,
    LRPHandler,
    SSVisualizer,
)
from lightning.pytorch.loggers import TensorBoardLogger
from omegaconf import OmegaConf
from tqdm import tqdm

from clearml import Logger, Task

# RESEARCH CODE
sys.path.append(os.path.join(os.getcwd()))
print(sys.path)

from detoxai.utils.experiment_logger import (  # noqa: E402
    ExperimentLogger,
)

from modules.research_code.src.models.pretrained import PretrainedModel  # noqa: E402

# --- GLOBAL CONFIG ---#
with open(os.path.join(os.getcwd(), "config", "global_config.json"), "r") as f:
    GCFG = json.load(f)
GCFG = OmegaConf.create(GCFG)
TASK_NAME = __file__.split("/")[-1].split(".")[0]


# --- ENVIROMENT SETUP ---#
torch.set_float32_matmul_precision("high")


def seed_everything(seed=42, full_determinism: bool = True):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    L.seed_everything(seed)
    if full_determinism:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def main(cfg):
    # -- Load configuration --
    config = OmegaConf.to_container(cfg, resolve=True)

    batch_size = config["experiment"]["dataloader"]["batch_size"]

    # dataset_config = config["dataset"]
    dataset_name = config["dataset"]["name"]
    device = config["detoxai"]["debias"]["methods_config"]["global"]["device"]
    enable_checkpointing = config["experiment"]["training"]["enable_checkpointing"]
    lr = config["experiment"]["training"]["lr"]
    max_epochs = config["experiment"]["training"]["max_epochs"]
    metrics = config["detoxai"]["debias"]["metrics"]
    methods_config = config["detoxai"]["debias"]["methods_config"]
    methods = config["detoxai"]["debias"]["methods"]
    model_name = config["model_name"]
    num_workers = config["experiment"]["dataloader"]["num_workers"]
    pareto_metrics = config["detoxai"]["debias"]["pareto_metrics"]
    prot_attr = config["protected_attribute"]
    prot_attr_val = config["protected_attribute_value"]
    return_type = config["detoxai"]["debias"]["return_type"]
    target = config["target_attribute"]

    variant = config["dataset"]["variant"]["name"]
    viz_batch = config["detoxai"]["visualization"]["viz_batch"]
    vis_max_images = config["detoxai"]["visualization"]["max_images"]
    vis_show_labels = config["detoxai"]["visualization"]["show_labels"]  # noqa
    # ----------------------------

    # -- Set logging --
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=config["general"]["logging_level"], force=True)
    logging.getLogger("matplotlib").setLevel(logging.INFO)
    logging.getLogger("PIL").setLevel(logging.INFO)
    logger.info(f"Config: {config}")
    logger.info(variant)
    # ----------------------------

    # -- ClearML task --
    task = Task.init(
        project_name=GCFG.project.name,
        task_name=TASK_NAME,
        task_type=Task.TaskTypes.training,
        reuse_last_task_id=False,
        tags=[model_name, dataset_name, f"y={target}", f"a={prot_attr}", variant],
        auto_resource_monitoring=False,  # thread safe
    )
    config = task.connect_configuration(config)
    queue_name = config["clearml"]["queue_name"]
    task.execute_remotely(queue_name=queue_name, exit_process=True, clone=False)
    # ----------------------------
    seed_everything(42)

    # -- Download datasets --
    from pathlib import Path

    detoxai_dataset_path = Path(os.environ["DETOXAI_DATASET_PATH"])

    if dataset_name in SUPPORTED_DATASETS:
        download_datasets([dataset_name])
    else:
        if "mnist" in dataset_name:
            if not os.path.exists(detoxai_dataset_path / dataset_name):
                logger.info("Unzipping MNIST dataset from resources")
                import shutil

                shutil.unpack_archive(
                    Path(os.getcwd()) / "resources" / f"{dataset_name}.zip",
                    Path(detoxai_dataset_path.parent),
                )
                # os.system(
                #     f"unzip ./resources/colored_mnist_artifact.zip -d {detoxai_dataset_path}"
                # )
            else:
                logger.info("MNIST dataset already exists, not unzipping")
        else:
            raise ValueError(f"Dataset {dataset_name} not supported")
    # ----------------------------

    # -- Create variant --

    variant_config = config["dataset"]["variant"]["variant_config"]

    variant_path = (
        Path(detoxai_dataset_path)
        / variant_config["dataset"]
        / "variants"
        / variant_config["variant"]
        / "splits"
    )
    if not variant_path.exists():
        logger.info(f"Variant {variant_config['variant']} does not exist")
        logger.info(f"Creating variant: {variant_config}")
        variant_path = make_detoxai_datasets_variant(variant_config)
    else:
        logger.info(
            f"Variant {variant_config['variant']} exists on path {variant_path}"
        )
    # ----------------------------

    # -- Get datasets --
    datasets: dict[str, DetoxaiDataset] = get_detoxai_datasets(
        config={
            "name": dataset_name,
            "target": target,
        },
        saved_variant=variant,
        transform=torchvision.transforms.Compose(
            [SquarePad(), torchvision.transforms.ToTensor()]  # resize_to=224
        ),
        device=device,
    )
    ds_train = datasets["train"]
    ds_test = datasets["test"]
    ds_unlearn = datasets["unlearn"]
    # ----------------------------

    # -- Log dataset splits --
    try:
        d = {}

        for k, v in datasets.items():
            d[f"{k} counts"] = (
                v.labels.loc[v.split_indices]
                .groupby([target, prot_attr])
                .count()["image_id"]
            )
            d[f"{k} distribution"] = v.labels.loc[v.split_indices].groupby(
                [target, prot_attr]
            ).count()["image_id"] / len(v.labels.loc[v.split_indices])

        df = pd.DataFrame(d).round(3)

        logger.info(f"Dataset splits description \n {df.T}")
    except Exception as e:
        logger.info(e)
    # ----------------------------

    # -- Create dataloaders --
    collate_fn = ds_train.get_collate_fn(prot_attr, prot_attr_val)

    seed_everything(42)
    dataloader_unlearn = DetoxaiDataLoader(
        ds_unlearn,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=True,
    )

    dataloader_test = DetoxaiDataLoader(
        ds_test,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=3,
    )

    seed_everything(42)
    dataloader_train = DetoxaiDataLoader(
        ds_train,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=3,
        shuffle=True,
    )
    # ----------------------------

    # -- Create model and trainer --
    class_cnt = ds_train.get_num_classes()

    x, _, _ = next(iter(dataloader_train))
    input_size = x.shape[1:]
    print("INPUT SIZE:", input_size)
    model = PretrainedModel(class_cnt, model_name).to(device)  # SimpleCNNv2
    # model = SimpleCNN(input_size, num_classes=class_cnt).to(device)
    # model = SimpleCNNv2(input_size, num_classes=class_cnt).to(device)

    tensorboard_logger = TensorBoardLogger(
        save_dir=f"{GCFG['path']['tensorboard_logs']}",
        name=f"{TASK_NAME}_{model_name}_{methods}",
    )

    trainer = L.Trainer(
        max_epochs=max_epochs,
        logger=tensorboard_logger,
        log_every_n_steps=10,
        enable_checkpointing=enable_checkpointing,
    )

    all_metrics = AllMetrics(
        DEFAULT_METRICS_CONFIG,
        class_labels=ds_train.get_class_names(),
        num_groups=2,
    )

    wrapped_model = FairnessLightningWrapper(
        model=model,
        performance_metrics=all_metrics.performance_metrics,
        fairness_metrics=all_metrics.fairness_metrics,
        learning_rate=lr,
    )
    seed_everything(42)
    trainer.fit(wrapped_model, dataloader_train)
    # ----------------------------

    torch.save(model, "~/model.pt")

    # -- Test --

    wrapped_model.eval()
    seed_everything(42)
    # trainer.test(wrapped_model, dataloader_test)
    # ----------

    # -- Log results to clearml --
    clearml_logger = Logger.current_logger()
    ex_logger = ExperimentLogger(clearml_logger)
    # ----------------------------

    # -- Visualize heatmaps on train and test sets --

    dataloader_unlearn_ss = DetoxaiDataLoader(
        ds_unlearn,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers // 3,
    )
    dataloader_test_ss = DetoxaiDataLoader(
        ds_test,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers // 3,
    )
    dataloader_train_ss = DetoxaiDataLoader(
        ds_train,
        collate_fn=collate_fn,
        batch_size=batch_size,
        num_workers=num_workers // 3,
    )

    dataloaders_zip = zip(  # noqa
        ["train", "test", "unlearn"],
        [dataloader_train_ss, dataloader_test_ss, dataloader_unlearn_ss],
    )

    # -- Debias -- #
    model = model.to(device)

    seed_everything(42)
    results: list[CorrectionResult] = debias(
        model=model,
        dataloader=dataloader_unlearn,
        methods=["ACLARC", "RRCLARC", "NT", "ZHANGM", "SAVANIAFT"],
        metrics=metrics,
        methods_config=methods_config,
        pareto_metrics=pareto_metrics,
        return_type=return_type,
        device=device,
        test_dataloader=dataloader_test,
    )

    x = config["xai"]["rectangle"]["x"]
    y = config["xai"]["rectangle"]["y"]
    w = config["xai"]["rectangle"]["w"]
    h = config["xai"]["rectangle"]["h"]

    rect_conf = {"rect": (x, y, w, h)}

    # -- Visualize heatmaps --
    for viz_batch in [0, 1, 2, 3]:
        for attr_meth in ["Gradient"]:
            for result in tqdm(results, desc="Visualizing heatmaps"):
                _model = result.get_model().to(device)
                _method_name = result.get_method()

                if attr_meth == "Gradient":
                    lrp_handler = LRPHandler(attributor_name=attr_meth)
                else:
                    lrp_handler = LRPHandler(
                        attributor_name=attr_meth, composite_name=None
                    )

                ss_visualizer = SSVisualizer(
                    dataloader_test,
                    _model,
                    lrp_handler,
                    draw_rectangles=True,
                    rectangle_config=rect_conf,
                )

                h_visualizer = HeatmapVisualizer(
                    dataloader_test,
                    _model,
                    lrp_handler,
                    draw_rectangles=True,
                    rectangle_config=rect_conf,
                )

                d_visualizer = DataVisualizer(
                    dataloader_test,
                    draw_rectangles=True,
                    rectangle_config=rect_conf,
                )

                ss_visualizer.attach_logger(ex_logger)
                h_visualizer.attach_logger(ex_logger)
                d_visualizer.attach_logger(ex_logger)

                ss_visualizer.visualize_batch(
                    viz_batch,
                    ConditionOn.PROPER_LABEL,
                    max_images=vis_max_images,
                    show_labels=False,
                )

                h_visualizer.visualize_batch(
                    viz_batch,
                    ConditionOn.PROPER_LABEL,
                    max_images=vis_max_images,
                )

                d_visualizer.visualize_batch(
                    viz_batch,
                    max_images=vis_max_images,
                    show_labels=False,
                )

                ss_visualizer.log(f"ss_{_method_name}_{attr_meth}_{viz_batch}", step=0)
                h_visualizer.log(f"h_{_method_name}_{attr_meth}_{viz_batch}", step=0)
                d_visualizer.log(f"d_{_method_name}_{attr_meth}_{viz_batch}", step=0)

                ss_visualizer.visualize_agg(viz_batch)
                h_visualizer.visualize_agg(viz_batch)
                d_visualizer.visualize_agg(viz_batch)

                ss_visualizer.log(
                    f"ss_{_method_name}_{attr_meth}_{viz_batch}_agg", step=0
                )
                h_visualizer.log(
                    f"h_{_method_name}_{attr_meth}_{viz_batch}_agg", step=0
                )
                d_visualizer.log(
                    f"d_{_method_name}_{attr_meth}_{viz_batch}_agg", step=0
                )

    # ----------------------------

    # -- Visuaslzie stuff
    batch_num = 0
    condition_on = ConditionOn.PROPER_LABEL.value

    lrp_handler = LRPHandler()
    result = lrp_handler.calculate(model, dataloader_test, batch_num=batch_num)
    _, labels, _ = dataloader_test.get_nth_batch(batch_num)  # noqa

    conditioned = []
    for i, label in enumerate(labels):
        # Assuming binary classification
        label = label if condition_on == ConditionOn.PROPER_LABEL.value else 1 - label
        conditioned.append(result[label, i])

    sailmaps: torch.Tensor = torch.stack(conditioned).to(dtype=float)
    sailmaps = sailmaps.cpu().detach().numpy()

    r = RRF()
    cut = r._sailmaps_rect(sailmaps, (33, 20), (150, 50))

    # plot sailmaps
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 4, figsize=(10, 5))
    for i in range(4):
        ax[i].imshow(cut[i])
        ax[i].axis("off")
    plt.suptitle("The rectangular cut of the sailmaps")

    # log as artifact
    ex_logger.log_image(fig, "sailmaps_rect", step=0)


if __name__ == "__main__":
    main()
