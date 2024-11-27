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
SEED=19
torch.manual_seed(3407)
torch.multiprocessing.set_sharing_strategy('file_system')
with open('./SNP_pca/1404cubic.pkl', 'rb') as f:
    x_data = pickle.load(f)
y1 = pd.read_csv("./SNP_pca/PH.Y")
y2 = pd.read_csv("./SNP_pca/KNPE.Y")
y3 = pd.read_csv("./SNP_pca/KWPE.Y")
y4 = pd.read_csv("./SNP_pca/DTA.Y")

data = pd.concat([x_data,y1, y2, y3, y4], axis=1)
train, test = train_test_split(data, test_size=0.999, random_state=SEED)

#'Hard', 'Prot'
col_names = data.columns.values.tolist()
  
label_cols = [ 'dta']



task_num = len(label_cols)
used_cols = [col for col in col_names if col not in ['ph', 'knpe','kwpe', 'dta']]
x_train, y_train = {name: train[name].values[:] for name in used_cols}, train[label_cols].values[:]
x_test, y_test = {name: test[name].values[:] for name in used_cols}, test[label_cols].values[:]

dg = DataGenerator(x_train, y_train)
_, _, test_dataloader = dg.generate_dataloader( x_val=x_test, y_val=y_test, 
                                      x_test=x_test, y_test=y_test, batch_size=640)
save_dir = './result/model.pth'
feature = [DenseFeature(col) for col in used_cols]#,"regression","regression"
task_types = ["regression"] * task_num
loaded_model = MMOE(feature, task_types, len(label_cols),
                    expert_params={"dims": [1024 * 4]},
                    tower_params_list=[{"dims": [1024 * 4]}] * len(label_cols))
loaded_model.load_state_dict(torch.load(save_dir))

mtl_trainer = MTLTrainer(loaded_model, task_types=task_types, optimizer_params={"lr": 0.1, "weight_decay": 0.11}, n_epoch=2, earlystop_patience=30, device=device, model_path=save_dir)


print("开始测试")
with torch.no_grad():
    auc = mtl_trainer.evaluate(mtl_trainer.model, test_dataloader)
print("预测准确度:", auc)