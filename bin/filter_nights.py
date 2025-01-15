import pandas as pd
from os import remove
from glob import glob

from naturalcogsci.helpers import get_project_root


PROJECT_ROOT =  get_project_root()

df = pd.read_csv(f"{PROJECT_ROOT}/data/nights/data.csv")
df = df[(df.votes >= 6) & (df.split == "test") & (~df.is_imagenet )].reset_index(drop=True)
images_to_keep = df.ref_path.to_list() + df.left_path.to_list() + df.right_path.to_list()
images_to_keep = [f"{PROJECT_ROOT}/dataset/nights/{x}" for x in images_to_keep]

all_files = glob(f"{PROJECT_ROOT}/data/nights/**/**/*png")
count = 0
for cur_file in all_files:
    if cur_file not in images_to_keep:
        remove(cur_file)