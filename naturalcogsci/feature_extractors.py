from __future__ import annotations
import glob
import os
from os.path import join
import json
from collections import OrderedDict

import numpy as np
import pandas as pd
from tqdm import tqdm
import torch
from PIL import Image
from sklearn.decomposition import PCA
import fasttext
from transformers import AutoTokenizer, AutoModel
from thingsvision import get_extractor, get_extractor_from_model
from thingsvision.utils.data import ImageDataset, DataLoader
import tensorflow_hub as hub

import tensorflow_hub as hub
import openai


from SLIP.models import (
    SLIP_VITS16,
    SLIP_VITB16,
    SLIP_VITL16,
    CLIP_VITS16,
    CLIP_VITB16,
    CLIP_VITL16,
    SIMCLR_VITS16,
    SIMCLR_VITB16,
    SIMCLR_VITL16,
)

from .helpers import get_project_root


def extract_features(
    feature_name: str,  # same as model name. In case different encoders are available, it is in `model_encoder` format
    use_cached: bool = True,  # If `True`, rerun extraction even if the features are saved. Defaults to True.
) -> None:
    """
    Extract features from a model and save to disk.
    """
    project_root = get_project_root()
    final_feature_path = join(
        project_root, "data", "features", f"{feature_name.replace('/', '_')}.npy"
    )

    hugging_face_dict = {
        "distilbert": "distilbert-base-uncased",
        "bert": "bert-base-uncased",
        "roberta": "roberta-base",
    }

    if os.path.exists(final_feature_path) and use_cached:
        return None

    if feature_name == "task":
        objects = folder_to_word(remove_digit_underscore=False)
        ids = pd.read_csv(join(project_root, "data", "THINGS", "unique_id.csv"))[
            "id"
        ].to_list()
        weights = np.loadtxt(
            join(project_root, "data", "THINGS", "spose_embedding_49d_sorted.txt")
        )

        features = [weights[ids.index(obj), :] for obj in objects]
        features = np.array(features)

    elif feature_name == "ada-002":
        openai.api_key = os.getenv("OPENAI_API_KEY")
        objects = folder_to_word(remove_digit_underscore=True)
        objects = [f"A photo of a {x}" for x in objects]
        features = np.array([get_ada_embedding(obj) for obj in objects])

    elif feature_name in ["bert", "roberta"]:
        objects = folder_to_word(remove_digit_underscore=True)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained(hugging_face_dict[feature_name])
        model = AutoModel.from_pretrained(hugging_face_dict[feature_name]).to(device)
        objects = [f"A photo of a {x}" for x in objects]

        tokenized_objects = tokenizer(
            objects, padding=True, truncation=True, return_tensors="pt"
        )
        tokenized_objects = {
            k: torch.tensor(v).to(device) for k, v in tokenized_objects.items()
        }

        with torch.no_grad():
            latent_objects = model(**tokenized_objects)

        features = latent_objects.last_hidden_state[:, 0, :].numpy()

    elif feature_name == "fasttext":
        objects = folder_to_word(remove_digit_underscore=True)
        ft = fasttext.load_model(
            join(
                project_root,
                "data",
                "embedding_weights_and_binaries",
                "crawl-300d-2M-subword.bin",
            )
        )
        features = [ft.get_word_vector(x) for x in objects]
        features = np.array(features)
    elif feature_name == "universal_sentence_encoder":
        objects = folder_to_word(remove_digit_underscore=True)
        objects = [f"A photo of a {x}" for x in objects]
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        model = hub.load(module_url)
        features = model(objects).numpy()

    elif feature_name == "pca":
        with open(join(project_root, "data", "features", "file_names.txt"), "r") as f:
            file_paths = [line.strip() for line in f]

        images = []
        for file_path in tqdm(file_paths):
            image = Image.open(file_path)
            image = image.resize((224, 224))
            images.append(np.array(image).flatten())

        data = np.stack(images)
        # to match the dimensionality of the generative features
        pca = PCA(n_components=49) 
        features = pca.fit_transform(data)

    elif "gLocal" in feature_name:
        # split feature_name by gLocal
        feature_name = feature_name.split("gLocal_")[-1]
        file_name = feature_name.replace("/", "_")
        features = np.load(join(project_root, "data", "features", f"{file_name}.npy"))
        glocal_transform = np.load(
            join(project_root, "data", "gLocal", f"{file_name}.npz")
        )

        features = (features - glocal_transform["mean"]) / glocal_transform["std"]
        features = features @ glocal_transform["weights"]

        if "bias" in glocal_transform:
            features += glocal_transform["bias"]

    else:
        features = get_visual_embedding(project_root, feature_name)

    feature_name = feature_name.replace("/", "_")
    np.save(final_feature_path, features)

    return None


def folder_to_word(remove_digit_underscore: bool,  # Remove digit and underscore from object names if true. Note that you need the digits to get the task embeddings, but not for the others. If True, the underscore gets replaced with a space.
) -> list:  # List of object names
    """
    Read file name directories and format them into words by parsing directories
    and, on demand, removing any numbers and underscores.
    """
    project_root = get_project_root()
    with open(join(project_root, "data", "features", "file_names.txt"), "r") as f:
        file_names = f.read()[:-1]  # there is an empty line in the end

    file_names = file_names.split("\n")
    file_names = [os.path.dirname(x) for x in file_names]
    file_names = [os.path.basename(x) for x in file_names]

    if remove_digit_underscore:
        file_names = ["".join([i for i in x if not i.isdigit()]) for x in file_names]
        file_names = [x.replace("_", " ") for x in file_names]
    return file_names


def get_visual_embedding(
    project_root: str,  # Root directory of the project
    feature_name: str,  # Name of the feature to extract. Must be in `model_config.json`
) -> np.ndarray:  # total images by features array
    """
    Extract visual embedding using `thingsvision`
    """

    pretrained = True

    slip_variants = {
        "slip_slip_small": SLIP_VITS16,
        "clip_slip_small": CLIP_VITS16,
        "simclr_slip_small": SIMCLR_VITS16,
        "slip_slip_base": SLIP_VITB16,
        "clip_slip_base": CLIP_VITB16,
        "simclr_slip_base": SIMCLR_VITB16,
        "slip_slip_large": SLIP_VITL16,
        "clip_slip_large": CLIP_VITL16,
        "simclr_slip_large": SIMCLR_VITL16,
    }

    with open(join(project_root, "data", "model_configs.json")) as f:
        file = json.load(f)
    model_config = file[feature_name]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_parameters = None
    save_name = feature_name
    if "slip" in feature_name:
        weights = torch.load(
            join(
                project_root,
                "data",
                "embedding_weights_and_binaries",
                f"{feature_name}.pth",
            ),
            map_location=device,
        )

        model = slip_variants[feature_name](
            ssl_mlp_dim=weights["args"].ssl_mlp_dim,
            ssl_emb_dim=weights["args"].ssl_emb_dim,
            rand_embed=False,
        )

        state_dict = OrderedDict()
        for k, v in weights["state_dict"].items():
            state_dict[k.replace("module.", "")] = v
        model.load_state_dict(state_dict, strict=True)
        model.eval()
        model.to(device)

        extractor = get_extractor_from_model(
            model=model, device=device, backend="pt", forward_fn=model.encode_image
        )

    else:
        if feature_name.startswith("clip"):
            model_parameters = {"variant": feature_name.split("clip_")[1]}
            save_name = feature_name
            feature_name = "clip"
        elif feature_name.startswith("OpenCLIP"):
            variant_name = feature_name.split("OpenCLIP_")[1].split("_laion")[0]
            dataset_name = feature_name.split(f"{variant_name}_")[1]
            model_parameters = {"variant": variant_name, "dataset": dataset_name}
            save_name = feature_name
            feature_name = "OpenCLIP"
        elif feature_name.startswith("Harmonization"):
            variant_name = feature_name.split("Harmonization_")[-1]
            model_parameters = {"variant": variant_name}
            save_name = feature_name
            feature_name = "Harmonization"
        elif feature_name.startswith("DreamSim"):
            variant_name = feature_name.split("DreamSim_")[1]
            model_parameters = {"variant": variant_name}
            save_name = feature_name
            feature_name = "DreamSim"

        save_name = save_name.replace("/", "_")
        extractor = get_extractor(
            model_name=feature_name,
            source=model_config["source"],
            device=device,
            pretrained=pretrained,
            model_parameters=model_parameters,
        )

    stimuli_path = join(project_root, "stimuli")
    batch_size = 1

    dataset = ImageDataset(
        root=stimuli_path,
        out_path=join(project_root, "data", "features"),
        backend=extractor.get_backend(),
        transforms=extractor.get_transformations(),
    )
    batches = DataLoader(
        dataset=dataset, batch_size=batch_size, backend=extractor.get_backend()
    )

    extractor.extract_features(
        batches=batches,
        module_name=model_config["module_name"],
        flatten_acts=False,
        output_dir=join(project_root, "data", "temp", save_name),
        step_size=1,
    )

    features = cleanup_temp(project_root, save_name)

    return features


def cleanup_temp(
    project_root: str,  # Root directory of the project
    save_name: str,  # name of the feature. has to match folder name under temp
) -> np.ndarray:  # total images by features array
    """
    Read features for single images from the temp folder.

    Combine them into one large array.
    """

    TOTAL_IMAGES = 26107

    temp_list = glob.glob(join(project_root, "data", "temp", save_name, "*npy"))
    temp_list_sorted = sorted(
        temp_list, key=lambda x: int("".join(filter(str.isdigit, x)))
    )

    # below we index into 0 for generality
    # it allows to extract the CLS from pytorch transformers
    # while having no effect on other embeddings, which are 1D vectors
    feature_array = np.array([np.load(x)[:,0, :] for x in temp_list_sorted])

    assert (
        feature_array.shape[0] == TOTAL_IMAGES
    ), f"There are features for only {feature_array.shape[0]} images.\nIt must be\
        {TOTAL_IMAGES} instead.\n temp won't be deleted and feature array won't be saved."

    return feature_array


def get_ada_embedding(
    text: str,  # Sentence to be embedded
    model: str = "text-embedding-ada-002",  # Model to get embeddings from. Defaults to "text-embedding-ada-002".
) -> np.ndarray:  # word vector
    """
    Generate word embeddings from openai ada model.
    """
    text = text.replace("\n", " ")
    return openai.Embedding.create(input=[text], model=model)["data"][0]["embedding"]
