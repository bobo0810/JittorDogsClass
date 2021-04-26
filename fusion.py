import sys
import glob
import yaml
import os
from net import BackboneFactory
import numpy as np
import json
from dataset.TestSet import TestDataset
from utils.tools import load_model_weights
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
yaml_dir='fusion.yaml'
#########################################

if __name__ == '__main__':
    # 读取conf
    file = open(os.path.join(cur_path+'/config/',yaml_dir), 'r',encoding="utf-8")
    test_conf = yaml.load(file, Loader=yaml.FullLoader)

    # 保存 多个模型的预测结果
    models_result=[]

    # 预测结果
    for i in range(len(test_conf['fusion'])):
        # 加载测试集
        test_loader = TestDataset(img_size= test_conf['fusion'][i]['img_size'],
                                  batch_size=test_conf['fusion'][i]['batch_size'],
                                  crop_type=test_conf['crop_type'])

        # 初始化网络
        backbone = BackboneFactory(test_conf['fusion'][i]['model'])
        # 加载权重
        writer = LogWriter() if jt.rank == 0 else None
        backbone = load_model_weights(backbone, test_conf['fusion'][i], writer)
        backbone.eval()

        result = []
        with jt.no_grad():
            for batch_idx, [imgdata, imgname] in enumerate(test_loader):
                outputs = backbone(imgdata)
                score = jt.nn.softmax(outputs, dim=1)
                for j in range(len(imgname)):
                    result.append([imgname[j], score[j].numpy()])
        models_result.append(result)

    # 融合
    json_result = {}
    for m in range(len(models_result[0])):
        # 图m的所有预测结果
        score_m = np.zeros(130)
        for n in range(len(models_result)):
            imgname,score = models_result[n][m]  #第n个模型 对应图m预测结果
            score_m +=score

        score_sort = np.argsort(-score_m)  # 降序排序的下标
        class_index = []
        # +1是提交时要求类别从1~130
        class_index.append(score_sort[0].item() + 1)
        class_index.append(score_sort[1].item() + 1)
        class_index.append(score_sort[2].item() + 1)
        class_index.append(score_sort[3].item() + 1)
        class_index.append(score_sort[4].item() + 1)
        json_result[imgname] = class_index
    re_js = json.dumps(json_result)
    fjson = open(cur_path+'/'+test_conf['save_name'], "w")
    print('json save in ' + cur_path+'/'+test_conf['save_name'])
    fjson.write(re_js)
    fjson.close()
