import sys
import glob
import yaml
from mode.train_batch import train
import os
import jittor as jt
if jt.has_cuda:
    jt.flags.use_cuda = 1
def add_project_path():
    rootpath = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(rootpath)
    sys.path.extend(glob.glob(rootpath + '/*'))
add_project_path()
cur_path = os.path.abspath(os.path.dirname(__file__))
#########################################
yaml_dir='train.yaml'
#########################################

if __name__ == '__main__':
    # 读取conf
    file = open(os.path.join(cur_path+'/config/',yaml_dir), 'r',encoding="utf-8")
    train_conf = yaml.load(file, Loader=yaml.FullLoader)
    train(train_conf)




