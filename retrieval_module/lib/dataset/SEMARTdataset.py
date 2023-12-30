import pandas as pd
import os
from torchvision.datasets import VisionDataset
from PIL import Image
from torchvision.transforms import transforms
from typing import Callable, Optional
from data_transform.transform_wrapper import TRANSFORMS
import unicodedata


class ImageTextDataset(VisionDataset):
    """
    Dtaset for loading image-text data for tasks like CLIP training, Image Captioning.

    Args:
        root: (string): The root path where the dataset is stored
        file_path: (string): Path to the file containing the image_paths/image_ids and associated captions.
            The expected format is csv with at least image_id, caption columns.
            `image_id`: id of the image.
            `caption`: caption of the image.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        name,
        cfg,
        root="",
        max_seq_length=70,
        preprocess = None,
        transform: Optional[Callable] = None
    ):
        super().__init__(root, transform)
        self.input_size = cfg.INPUT_SIZE
        self.cfg = cfg
        self.mode = name
        self.transform = transform
        self.preprocess = preprocess
        self.update_transform()


        if self.mode == "train":
            file_path = cfg.DATASET.TRAIN_CSV
        elif self.mode == "eval" or name == "val":
            file_path = cfg.DATASET.VALID_CSV
        elif self.mode == "test":
            file_path = cfg.DATASET.TEST_CSV
        else:
            raise ValueError(f"{name} dataset is not supported!")

        if not os.path.exists(file_path):
            file_path = os.path.join(cfg.DATASET.DATA_DIR, file_path)

        image_dir = cfg.DATASET.IMAGE_DIR

        df = pd.read_csv(file_path, sep='\t', encoding = "cp1252")
        self.captions = df['DESCRIPTION'].tolist()
        self.image_names = df['IMAGE_FILE'].tolist()
        self.image_paths = df['IMAGE_FILE'].apply(lambda x: os.path.join(image_dir, x)).tolist()
        self.max_seq_length = cfg.TRAIN.MAX_SEQ_LENGTH

        self.captions = [unicodedata.normalize("NFKD", c) for c in self.captions]


    def _load_image(self, idx: int):
        path = self.image_paths[idx]
        image = Image.open(path)
        if self.preprocess:
            image = self.preprocess(image)
        return image

    def _load_target(self, idx):
        return self.captions[idx]

    def __getitem__(self, index: int):
        image = self._load_image(index)
        target = self._load_target(index)
        image_name = self.image_names[index]

        return index, image_name, image, target

    def __len__(self) -> int:
        return len(self.captions)

    def update_transform(self, input_size=None):
        normalize = TRANSFORMS["normalize"](cfg=self.cfg, input_size=input_size)
        transform_list = [transforms.ToPILImage()]
        transform_ops = (
                self.cfg.TRANSFORMS.TRAIN_TRANSFORMS
                if self.mode == "train"
                else self.cfg.TRANSFORMS.TEST_TRANSFORMS
                )
        for tran in transform_ops:
            transform_list.append(TRANSFORMS[tran](cfg=self.cfg, input_size=input_size))
        transform_list.extend([transforms.ToTensor(), normalize])
        self.transform = transforms.Compose(transform_list)
