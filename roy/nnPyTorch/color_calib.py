import os
import pandas as pd

PLANET_KAGGLE_ROOT = os.path.abspath("../../input/")
PLANET_KAGGLE_JPEG_DIR = os.path.join(PLANET_KAGGLE_ROOT, 'train-jpg')
PLANET_KAGGLE_LABEL_CSV = os.path.join(PLANET_KAGGLE_ROOT, 'train_v2.csv')
assert os.path.exists(PLANET_KAGGLE_ROOT)
assert os.path.exists(PLANET_KAGGLE_JPEG_DIR)
assert os.path.exists(PLANET_KAGGLE_LABEL_CSV)


labels_df = pd.read_csv(PLANET_KAGGLE_LABEL_CSV)



from dataset.kgforest import KgForestDataset

dataset = KgForestDataset('testv2-20522', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                            transform=[
                                lambda x: img_to_tensor(x),
                            ],
                            width=256,height=256,
                            ext='all',
                            is_preload=False,
                            label_csv=None,
                          )

dataset.sample(100)
