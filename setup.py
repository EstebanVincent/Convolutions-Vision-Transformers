import kaggle
from PIL import Image
import os
import shutil
import pandas as pd


def load():
    kaggle.api.authenticate()
    kaggle.api.dataset_download_files(
        "balabaskar/tom-and-jerry-image-classification", path='./', unzip=True)


def architecture(size):
    src_dir = "tom_and_jerry"   # 1280*720 and 854*480
    dst_dir = "dataset/imgs"    # size*size

    if not os.path.exists(dst_dir):
        os.makedirs(dst_dir)

    for subdir in ["jerry", "tom", "tom_jerry_0", "tom_jerry_1"]:
        src_subdir = os.path.join(src_dir, "tom_and_jerry", subdir)

        for file in os.listdir(src_subdir):
            src_file = os.path.join(src_subdir, file)
            dst_file = os.path.join(dst_dir, file)

            if os.path.isfile(src_file):
                image = Image.open(src_file)
                image = image.resize((size, size), Image.ANTIALIAS)
                image.save(dst_file)

    print("Folder created successfully")

    shutil.rmtree(src_dir)
    print("Folder deleted successfully")


def csv():
    df = pd.read_csv("ground_truth.csv")
    df["class"] = 2 * df["tom"] + df["jerry"]
    # (tom = 0, jerry = 0) -> class = 0
    # (tom = 0, jerry = 1) -> class = 1
    # (tom = 1, jerry = 0) -> class = 2
    # (tom = 1, jerry = 1) -> class = 3

    df.to_csv("dataset/ground_truth.csv", index=False)

    os.remove("ground_truth.csv")
    print("File deleted successfully")

    os.remove("challenges.csv")
    print("File deleted successfully")


def setup(size=128):
    load()
    architecture(size)
    csv()
