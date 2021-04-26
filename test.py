import sys
import glob
import yaml
import os
import json
from net import BackboneFactory
from dataset.TestSet import TestDataset
from utils.tools import load_model_weights
import time
import jittor as jt
from visualdl import LogWriter
jt.flags.use_cuda = 1 # 1:True启动GPU
def add_project_path():
    rootpath = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(rootpath)
    sys.path.extend(glob.glob(rootpath + '/*'))
add_project_path()
cur_path = os.path.abspath(os.path.dirname(__file__))
#########################################
yaml_dir='test.yaml'
#########################################

if __name__ == '__main__':
    # 读取conf
    file = open(os.path.join(cur_path+'/config/',yaml_dir), 'r',encoding="utf-8")
    test_conf = yaml.load(file, Loader=yaml.FullLoader)

    localtime = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())


    writer=None
    if jt.rank == 0:
        writer = LogWriter()

    # 初始化网络和分类器
    backbone = BackboneFactory(test_conf['net']['name'])

    # 加载权重
    backbone= load_model_weights(backbone, test_conf['net'], writer)
    backbone.eval()

    # 加载测试集
    test_loader = TestDataset(img_size=test_conf['img_size'],
                              batch_size=test_conf['batch_size'],
                              crop_type=test_conf['crop_type'])

    json_result = {}
    for batch_idx, [imgdata, imgname] in enumerate(test_loader):

        with jt.no_grad():
            outputs=backbone(imgdata)
            index, value=jt.argsort(outputs,descending=True, dim=1) # 预测概率降序排列
        # 写入json
        for i in range(len(imgname)):
            class_index = []
            # +1是提交时要求类别从1~130
            class_index.append(index[i][0].item()+1)
            class_index.append(index[i][1].item()+1)
            class_index.append(index[i][2].item()+1)
            class_index.append(index[i][3].item()+1)
            class_index.append(index[i][4].item()+1)
            json_result[imgname[i]] = class_index
    re_js = json.dumps(json_result)
    fjson = open(cur_path+'/'+test_conf['save_name'], "w")
    print('json save in ' + cur_path + '/' + test_conf['save_name'])
    fjson.write(re_js)
    fjson.close()
