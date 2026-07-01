#!/home/charles/panda/panda_env310/bin/python3.10

import sys
from pathlib import Path

# Add the root directory to sys.path
sys.path.append("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts")

import os
from pathlib import Path
import numpy as np
import shutil
import json
import matplotlib.pyplot as plt 
import cv2
import torch
import torchvision
import numpy as np
from pathlib import Path
import typing
from PIL import Image
from utili.recap import URI, CfgNode as CN

from chessrec.preprocessing import gen_occ_data, gen_pie_data
from chessrec.core.dataset import build_dataset, build_data_loader, Datasets, unnormalize
from chessrec.core.models import build_model
from chessrec.core.statistics import StatsAggregator
from chessrec.core import device, DEVICE

def clean_folder(folder):
    if not os.path.exists(folder):
        os.mkdir(folder)
    else:
        for filename in os.listdir(folder):
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')

def _csv(model: torch.nn.Module, agg: StatsAggregator, model_name: str, mode: Datasets) -> str:
    params = sum([np.prod(p.size()) for p in model.parameters()])
    return ",".join(map(str, [model_name,
                              mode.value,
                              params,
                              agg.accuracy(),
                              *map(agg.precision, agg.classes),
                              *map(agg.recall, agg.classes),
                              *map(agg.f1_score, agg.classes),
                              *agg.confusion_matrix.flatten()
                              ]))


def _csv_heading(classes: typing.List[str]) -> str:
    def class_headings(metric: str) -> typing.List[str]:
        return [f"{metric}/{c}" for c in classes]
    return ",".join(["model",
                     "dataset",
                     "parameters",
                     "accuracy",
                     *class_headings("precision"),
                     *class_headings("recall"),
                     *class_headings("f1_score"),
                     *(f"confusion_matrix/{i}/{j}"
                       for i in range(len(classes))
                       for j in range(len(classes)))])


def evaluate(model_path: Path, datasets: typing.List[Datasets], output_folder: Path, find_mistakes: bool = False, include_heading: bool = False) -> str:
    """Evaluate a model, returning the results as CSV.

    Args:
        model_path (Path): path to the model file
        datasets (typing.List[Datasets]): the datasets to evaluate on
        output_folder (Path): output folder for the mistake images (if applicable)
        find_mistakes (bool, optional): whether to output all mistakes as images to the output folder. Defaults to False.
        include_heading (bool, optional): whether to include a heading in the CSV output. Defaults to False.

    Raises:
        ValueError: if the YAML config file is missing

    Returns:
        str: the CSV string
    """
    model_file = model_path
    yaml_file = next(iter(model_path.parent.glob("*.yaml")))
    print(f"current model: {model_file}")
    model_name = model_file.stem
    cfg = CN.load_yaml_with_base(yaml_file)
    model = build_model(cfg)
    model.load_state_dict(torch.load(model_file, map_location=DEVICE))
    model = device(model)
    model.eval()

    datasets = {mode: build_dataset(cfg, mode)
                for mode in datasets}
    classes = next(iter(datasets.values())).classes

    csv = []
    if include_heading:
        csv.append(_csv_heading(classes))
    for mode, dataset in datasets.items():
        # Load dataset
        loader = build_data_loader(cfg, dataset, mode)
        # Compute statistics over whole dataset
        agg = StatsAggregator(classes)
        for images, labels in device(loader):
            predictions = model(images)
            agg.add_batch(predictions, labels, **(dict(inputs=images)
                                                  if find_mistakes else dict()))

        csv.append(_csv(model, agg, model_name, mode))
        if find_mistakes:
            predicted_mistakes = zip(*sorted(agg.mistakes,
                                                key=lambda x: x[0]))
            if len(list(predicted_mistakes)) != 0:
                groundtruth, wrongclass, mistakes = zip(*sorted(agg.mistakes,
                                                key=lambda x: x[0]))
                imgs = torch.tensor(np.array(mistakes)).permute((0, 2, 3, 1))
                imgs = unnormalize(imgs).permute((0, 3, 1, 2))
                # imgs = imgs.permute((0, 3, 1, 2))
                img = torchvision.utils.make_grid(imgs, pad_value=1, nrow=4)
                img = img.numpy().transpose((1, 2, 0)) * 255
                img = Image.fromarray(img.astype(np.uint8))
                mistakes_file = output_folder / \
                    f"{model_name}_{mode.value}_mistakes.png"
                img.save(mistakes_file)
                groundtruth_file = output_folder / \
                    f"{model_name}_{mode.value}_groundtruth.csv"
                with groundtruth_file.open("w") as f:
                    f.write(",".join(map(str, groundtruth)) + "\n")
                    f.write(",".join(map(str, wrongclass)))
            else:
                groundtruth_file = output_folder / \
                    f"{model_name}_{mode.value}_no_mistake.txt"
                with groundtruth_file.open("w") as f:
                    f.write(f"predicted_mistakes: {predicted_mistakes}")

        # del model
        torch.cuda.empty_cache()
    return "\n".join(csv)

from chessrec.classifier import occ_models, piece_models

#  Evaluate the selected models
best_model_folder = Path(f"/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/chessrec/runs/best_models")
output_folder = Path("/home/charles/panda/catkin_ws/src/move_chess_panda/scripts/data/evaluation/best_models/shifted")
clean_folder(output_folder)
classifiers = ["piece_classifier"]
datasets = (Datasets.SHIFTED, Datasets.SHIFTED)
find_mistakes = False

output_folder.mkdir(parents=True, exist_ok=True)
for classifier in classifiers:
    subfolder = output_folder / classifier 
    clean_folder(subfolder)
    # subfolder.mkdir(parents=True, exist_ok=True)
    output_csv = subfolder / "evaluation.csv"
    with output_csv.open("w") as f:
        models = list((best_model_folder).glob("**/*.pt"))
        for model in models:
            model.rename(model.parent / f"{model.parent.parent.stem}_{model.stem[-8:]}.pt")
        models = list((best_model_folder).glob("**/*.pt"))
        for i, model in enumerate(models):
            print(f"Processing {classifier} model {i+1}/{len(models)}")
            with torch.no_grad():
                f.write(evaluate(model, datasets, output_folder,
                                find_mistakes=find_mistakes,
                                include_heading=i == 0) + "\n")