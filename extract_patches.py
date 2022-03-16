"""extract_patches.py

Patch extraction script.
"""

import re
import glob
import os
import tqdm
import pathlib
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

import numpy as np

from misc.patch_extractor import PatchExtractor
from misc.utils import rm_n_mkdir
import joblib

from dataset import get_dataset

# -------------------------------------------------------------------------------------
if __name__ == "__main__":

    # Determines whether to extract type map (only applicable to datasets with class labels).
    type_classification = True

    win_size = [270, 270]
    step_size = [256, 256]
    extract_type = "mirror"  # Choose 'mirror' or 'valid'. 'mirror'- use padding at borders. 'valid'- only extract from valid regions.

    # Name of dataset - use Kumar, CPM17, CoNSeP or conic.
    # This used to get the specific dataset img and ann loading scheme from dataset.py
    dataset_name = "conic"

    assert dataset_name in  ["consep","kumar","cpm17", "conic"]

    save_root = "./dataset/%s/" % dataset_name

    # a dictionary to specify where the dataset path should be changed.
    
    if dataset_name == "conic":
        assert win_size == [270, 270] and step_size == [256, 256] and extract_type == "mirror" #conic size
        #generating train/test splits and choose the best one
        info = pd.read_csv(f'./data/conic/patch_info.csv') #name of each images
        images_conic = np.load(f'./data/conic/images.npy')
        labels_conic = np.load(f'./data/conic/labels.npy')

        file_names = np.squeeze(info.to_numpy()).tolist()

        img_sources = [v.split('-')[0] for v in file_names]
        img_sources = np.unique(img_sources)

        cohort_sources = [v.split('_')[0] for v in img_sources]
        _, cohort_sources = np.unique(cohort_sources, return_inverse=True)

        train_size = 0.8

        splitter = StratifiedShuffleSplit(
        n_splits=10,
        train_size=train_size,
        test_size=1-train_size,
        random_state=0
        )

        splits = []
        split_generator = splitter.split(img_sources, cohort_sources)
        for train_indices, valid_indices in split_generator:
            train_cohorts = img_sources[train_indices]
            valid_cohorts = img_sources[valid_indices]
            assert np.intersect1d(train_cohorts, valid_cohorts).size == 0
            train_names = [
                file_name
                for file_name in file_names
                for source in train_cohorts
                if source == file_name.split('-')[0]
            ]
            valid_names = [
                file_name
                for file_name in file_names
                for source in valid_cohorts
                if source == file_name.split('-')[0]
            ]
            train_names = np.unique(train_names)
            valid_names = np.unique(valid_names)
            assert np.intersect1d(train_names, valid_names).size == 0
            train_indices = [file_names.index(v) for v in train_names]
            valid_indices = [file_names.index(v) for v in valid_names]
            splits.append([train_indices, valid_indices])

        idx = np.array([np.abs(len(i[0])/(len(i[0])+len(i[1])) - train_size) for i in splits]).argmin()
        train_indices = splits[idx][0]
        valid_indices = splits[idx][1]
        splits = splits[idx]
        joblib.dump(splits, f"./data/conic/split.dat") #dumping split information

        #use the best split, apply transformation and create individual .npys

        #TODO: implement progress bar

        for c1,idxs in enumerate(splits):
            out_dir = "./%s/%s/%s/%dx%d_%dx%d/" % (
                    save_root,
                    dataset_name,
                    "train" if c1==0 else "valid",
                    win_size[0],
                    win_size[1],
                    step_size[0],
                    step_size[1],
                )
            rm_n_mkdir(out_dir)
            for c,idx in enumerate(idxs):

                img = images_conic[idx]
                label = labels_conic[idx]
                img_label = np.concatenate([img, label], axis=2)
                img_label = np.pad(img_label,((7,),(7,),(0,)),mode="reflect") #256x256 pad to 270x270
                np.save("{0}/{1}_{2:03d}.npy".format(str(out_dir), "train" if c1==0 else "valid", c), img_label)

    else:
        # consep, change the path appropriately if decided to use kumar etc
        dataset_info = {
            "train": {
                "img": (".png", "./data/CoNSeP/Train/Images"),
                "ann": (".mat", "./data/CoNSeP/Train/Labels"),
            },
            "valid": {
                "img": (".png", "./data/CoNSeP/Test/Images"),
                "ann": (".mat", "./data/CoNSeP/Test/Labels"),
            },
        }

        patterning = lambda x: re.sub("([\[\]])", "[\\1]", x)
        parser = get_dataset(dataset_name)
        xtractor = PatchExtractor(win_size, step_size)
        for split_name, split_desc in dataset_info.items():
            img_ext, img_dir = split_desc["img"]
            ann_ext, ann_dir = split_desc["ann"]

            out_dir = "./%s/%s/%s/%dx%d_%dx%d/" % (
                save_root,
                dataset_name,
                split_name,
                win_size[0],
                win_size[1],
                step_size[0],
                step_size[1],
            )
            file_list = glob.glob(patterning("%s/*%s" % (ann_dir, ann_ext)))
            file_list.sort()  # ensure same ordering across platform

            rm_n_mkdir(out_dir)

            pbar_format = "Process File: |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
            pbarx = tqdm.tqdm(
                total=len(file_list), bar_format=pbar_format, ascii=True, position=0
            )

            for file_idx, file_path in enumerate(file_list):
                base_name = pathlib.Path(file_path).stem

                img = parser.load_img("%s/%s%s" % (img_dir, base_name, img_ext))
                ann = parser.load_ann(
                    "%s/%s%s" % (ann_dir, base_name, ann_ext), type_classification
                )

                # *
                img = np.concatenate([img, ann], axis=-1)
                sub_patches = xtractor.extract(img, extract_type)

                pbar_format = "Extracting  : |{bar}| {n_fmt}/{total_fmt}[{elapsed}<{remaining},{rate_fmt}]"
                pbar = tqdm.tqdm(
                    total=len(sub_patches),
                    leave=False,
                    bar_format=pbar_format,
                    ascii=True,
                    position=1,
                )

                for idx, patch in enumerate(sub_patches):
                    np.save("{0}/{1}_{2:03d}.npy".format(out_dir, base_name, idx), patch)
                    pbar.update()
                pbar.close()
                # *

                pbarx.update()
            pbarx.close()
