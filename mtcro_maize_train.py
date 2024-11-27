from torch_rechub.models.multi_task import MMOE
import torch
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch_rechub.trainers import MTLTrainer
from torch_rechub.basic.features import DenseFeature
from torch_rechub.utils.data import DataGenerator
from scipy.stats import pearsonr
import sys
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("*"*80)
SEED=19
print("SEED: ",SEED)
#kwpe:seed2
#ph:seed1
#dta:seed3
torch.manual_seed(3407)
torch.multiprocessing.set_sharing_strategy('file_system')
with open('./SNP_pca/1404cubic.pkl', 'rb') as f:
    x_data = pickle.load(f)
y1 = pd.read_csv("./SNP_pca/PH.Y")
y2 = pd.read_csv("./SNP_pca/KNPE.Y")
y3 = pd.read_csv("./SNP_pca/KWPE.Y")
y4 = pd.read_csv("./SNP_pca/DTA.Y")

data = pd.concat([x_data,y1, y2, y3, y4], axis=1)
print(data.shape)
train, test = train_test_split(data, test_size=0.1, random_state=SEED)

#'Hard', 'Prot'
col_names = data.columns.values.tolist()
print(data.shape)
label_cols = [ 'dta']



task_num = len(label_cols)
used_cols = [col for col in col_names if col not in ['ph', 'knpe','kwpe', 'dta']]
x_train, y_train = {name: train[name].values[:] for name in used_cols}, train[label_cols].values[:]
x_test, y_test = {name: test[name].values[:] for name in used_cols}, test[label_cols].values[:]

dg = DataGenerator(x_train, y_train)
train_dataloader, val_dataloader, test_dataloader = dg.generate_dataloader( x_val=x_test, y_val=y_test, 
                                      x_test=x_test, y_test=y_test, batch_size=640)

feature = [DenseFeature(col) for col in used_cols]#,"regression","regression"
task_types = ["regression"] * task_num
model = MMOE(feature, task_types, task_num, expert_params={"dims": [1024*4],"dims": [1024*4],"dims": [1024*4],\
                                                    "dims": [1024*4]}, tower_params_list=[{"dims": [1024*4]}]* task_num)

#相关参数、训练模型及评估
learning_rate = 1e-4
weight_decay = 1e-5
epoch = 100
save_dir = './result/model.pth'
mtl_trainer = MTLTrainer(model, task_types=task_types, optimizer_params={"lr": learning_rate, "weight_decay": weight_decay}, n_epoch=epoch, earlystop_patience=30, device=device, model_path=save_dir)
mtl_trainer.fit(train_dataloader, val_dataloader)

torch.save(mtl_trainer.model.state_dict(),save_dir)
print("Begin test")
auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
print(auc)