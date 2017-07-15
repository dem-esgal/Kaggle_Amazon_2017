#from net.common import *
#from net.dataset.tool import *
import sys
import skimage.io
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import *


TOP_DIR ='../../input/'
TRAIN_JPG_DIR = TOP_DIR+'train-jpg/'
TRAIN_TIF_DIR = TOP_DIR+'train-tif-v2/'
TEST_JPG_DIR = TOP_DIR+'test-jpg-additional/'
TEST_TIF_DIR = TOP_DIR+'test-tif-v2/'
LABEL_CSV = TOP_DIR+'train_v2.csv'



CLASS_NAMES=[
    'clear',    	 # 0
    'haze',	         # 1
    'partly_cloudy', # 2
    'cloudy',	     # 3
    'primary',	     # 4
    'agriculture',   # 5
    'water',	     # 6
    'cultivation',	 # 7
    'habitation',	 # 8
    'road',	         # 9
    'slash_burn',	 # 10
    'conventional_mine', # 11
    'bare_ground',	     # 12
    'artisinal_mine',	 # 13
    'blooming',	         # 14
    'selective_logging', # 15
    'blow_down',	     # 16
]

label_df = pd.read_csv(LABEL_CSV)
tmp = label_df.tags.str.get_dummies(sep=" ")
label_df = pd.concat( [label_df, tmp[CLASS_NAMES] ], axis=1)
label_df = label_df.set_index( label_df.image_name)


# helper functions -------------
def score_to_class_names(prob, class_names, threshold = 0.5, nil=''):

    N = len(class_names)
    if not isinstance(threshold,(list, tuple, np.ndarray)) : threshold = [threshold]*N

    s=nil
    for n in range(N):
        if prob[n]>threshold[n]:
            if s==nil:
                s = class_names[n]
            else:
                s = '%s %s'%(s, class_names[n])
    return s

## custom data loader -----------------------------------

def load_one_image(imgkey, width, height, ext):

    if ext =='tif':
        scale = 65536
        if pd.isnull(imgkey.tags):
            img_file = TEST_TIF_DIR + imgkey.image_name + ".tif"
        else:
            img_file = TRAIN_TIF_DIR + imgkey.image_name + ".tif"
        img_file = skimage.io.imread(img_file)
        img_file = img_file[:,:,(3,2,1,0)] #B G R IR to IR R G B

    elif ext =='jpg':
        scale = 256
        if pd.isnull(imgkey.tags):
            img_file = TEST_JPG_DIR + imgkey.image_name + ".jpg"
        else:
            img_file = TRAIN_JPG_DIR + imgkey.image_name + ".jpg"
        img_file = skimage.io.imread(img_file)

    h,w = img_file.shape[0:2]
    if height!=h or width!=w:
        img_file = cv2.resize(img_file,(height,width))

    img_file = img_file.astype(np.float32)
    img_file /= scale

    return img_file

def tif_color_corr(img):
    # these vals are from run_fit()
    means = [0.097610317, 0.047554102, 0.065722242, 0.076658122]
    stds = [0.028613893, 0.026216319, 0.025296433, 0.028521383]

    img -= means
    img /= stds
    img = img*0.125 + 0.5 #scale +- 2SD to (0,1)
    return img

class KgForestDataset(Dataset):

    def __init__(self, keylist, transform=None, height=256, width=256, outfields=['jpg'], cacheGB=0):
        """
        cacheGB: in GB, 0 means off
        """
        print("init", self)
        with open(keylist) as f:
            keys = f.readlines()
        keys = [x.strip()for x in keys]
        self.num = len(keys)
        try:
            self.df = label_df.loc[keys]
        except Exception as e:
            self.df = pd.DataFrame(index = keys)
            self.df['image_name'] = keys
            self.df['tags'] = np.nan

        self.transform = transform
        self.outfields = outfields

        self.cacheSize = cacheGB*1000*1024*1024
        self.cacheUsed = 0
        self.cacheDict = {}

        self.width = width
        self.height = height


    #https://discuss.pytorch.org/t/trying-to-iterate-through-my-custom-dataset/1909
    def __getitem__(self, index):
        #print ('\tcalling Dataset:__getitem__ @ idx=%d'%index)

        if self.cacheSize>0 and index in self.cacheDict:
            #print('cache used')
            return self.cacheDict[index]

        output = {}
        for outfield in self.outfields:
            if outfield == 'jpg':
                result = load_one_image(self.df.iloc[index], self.width, self.height, 'jpg')
            if outfield == 'tif':
                result = load_one_image(self.df.iloc[index], self.width, self.height, 'tif')
            if outfield == 'label':
                result = self.df[CLASS_NAMES].iloc[index].values

            if self.transform is not None:
                if outfield in ['tif','jpg']:
                    for t in self.transform:
                        result = t(result)

            output[outfield] = result

        totSize = 0
        for aItem in output:
            totSize += sys.getsizeof(aItem)

        if self.cacheUsed < self.cacheSize:
            self.cacheDict[index] = output
            self.cacheUsed += totSize
            #print('cache entries', len(self.cacheDict))

        return output


    def __len__(self):
        #print ('\tcalling Dataset:__len__')
        return len(self.df)





# fit sigmoid curve for displaying tif
def run_fit():

    nSamples = 2560

    dataset = KgForestDataset('labeled.txt', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                width=256,height=256,
                                outfields = ['tif', 'label'],
                                cacheGB=-1,
                              )

    samples = np.stack((dataset[i]['tif'] for i in range(nSamples)))
    means = [samples[:,:,:,i].mean() for i in range(4)]
    stds = [samples[:,:,:,i].std() for i in range(4)]
    print(means)
    print(stds)

def plot_img(img):
    import matplotlib.pyplot as plt

    fig = plt.figure()
    fig.set_size_inches(8, 4)
    img = np.clip(img, 0,1)

    if img.shape[2]==3:
        a = fig.add_subplot(1, 2, 1)
        a.set_title('R-G-B')
        plt.imshow(img[:,:,(0,1,2)]) #JPG RGB

    if img.shape[2]==4:
        a = fig.add_subplot(1, 2, 1)
        a.set_title('R-G-B')
        plt.imshow(img[:,:,(1,2,3)]) #TIF RGB

        a = fig.add_subplot(1, 2, 2)
        a.set_title('IR-R-G')
        plt.imshow(img[:,:,(0,1,2)]) #TIF IR,R,G as RGB

    plt.show()


#test dataset
def run_check_dataset():
    dataset = KgForestDataset('val_5000.txt', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                width=256,height=256,
                                transform=[
                                    tif_color_corr,
                                ],
                                outfields = ['tif', 'label'],
                                cacheGB=5,
                              )
    sampler = SequentialSampler(dataset)  #FixedSampler(dataset,[5,]*100  )    #RandomSampler
    loader  = DataLoader(dataset, batch_size=32, sampler=sampler,  drop_last=False, pin_memory=True,
                         num_workers = 0,
                        )

    for epoch in range(10):
        print('epoch=%d -------------------------'%(epoch))
        for (batchID, batch) in enumerate(loader):
            #means = [batch['tif'][:,:,:,i].mean() for i in range(4)]
            #print(batchID, means)
            print(batchID)

    print('sucess')

# main #################################################################
if __name__ == '__main__':
    import os
    print( '%s: calling main function ... ' % os.path.basename(__file__))

    #run_check_dataset()
    #run_fit()
    #run_check_dataset()

    dataset_jpg_plain = KgForestDataset('labeled.txt', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                width=256,height=256,
                                outfields = ['jpg', 'label'],
                                cacheGB=-1,
                              )

    dataset_tif_plain = KgForestDataset('labeled.txt', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                width=256,height=256,
                                outfields = ['tif', 'label'],
                                cacheGB=-1,
                              )

    dataset_tif_corr = KgForestDataset('labeled.txt', #'valid-8000', ##'debug-32', ###'train-40479',  ##'train-ordered-20', ##
                                width=256,height=256,
                                transform=[
                                    tif_color_corr,
                                ],
                                outfields = ['tif', 'label'],
                                cacheGB=-1,
                              )

    def inspec(i):
        out = dataset_tif_corr[i]
        tags = [CLASS_NAMES[i] for i in np.where(out['label']==1)[0]]
        print(" ".join(tags))
        plot_img(out['tif'])
