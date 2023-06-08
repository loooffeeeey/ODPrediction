import os
import sys
import json
import random
import datetime

import numpy as np

import torch

from pprint import pprint

def gpu_info(mem_need = 10000):
    gpu_status = os.popen('nvidia-smi | grep %').read().split('|')

    mem_idx = [2, 6, 10, 14, 18, 22, 26, 30]
    mem_bus = [x for x in range(8)]
    mem_list = []
    for idx, info in enumerate(gpu_status):
        if idx in mem_idx:
            mem_list.append(11019 - int(info.split('/')[0].split('M')[0].strip()))
    idx = np.array(mem_bus).reshape([-1, 1])
    mem = np.array(mem_list).reshape([-1, 1])
    id_mem = np.concatenate((idx, mem), axis=1)
    GPU_available = id_mem[id_mem[:,1] >= mem_need][:,0]

    if len(GPU_available) != 0:
        return GPU_available
    else:
        return None

def narrow_setup(interval = 0.5, mem_need = 10000):
    GPU_available = gpu_info()
    i = 0
    while GPU_available is None:  # set waiting condition
        GPU_available = gpu_info(mem_need)
        i = i % 5
        symbol = 'monitoring: ' + '>' * i + ' ' * (10 - i - 1) + '|'
        sys.stdout.write('\r' + ' ' + symbol)
        sys.stdout.flush()
        # time.sleep(interval)
        i += 1
    GPU_selected = random.choice(GPU_available)
    return GPU_selected

def get_conifg(path):
    config = json.load(open(path, "r"))
    print("\n****** experiment name:", config["exp_name"], " ******")

    if config["batch_size"] <= 1:
        raise Exception("Batch size cannot be set as 1 or less.")

    # 实验管理
    if config["mode"] == "init":
        timestamp = str(datetime.datetime.now()).replace("-", "").split(".")[0].replace(" ", "T")[:-3].replace(":","")
        config["exp_name"] = config["exp_name"] + timestamp
    else:
        # 指定要load的实验的时间戳
        saved_model_timestamp = config["load_timestamp"]
        config["exp_name"] = config["exp_name"] + saved_model_timestamp
        config.pop("load_timestamp")

    # check GPU available
    if config["check_device"] == 1:
        GPU_no = narrow_setup(interval = 1, mem_need = 8100)
        config["device"] = torch.device(int(GPU_no))
        print("\n****** Using No.", int(GPU_no), "GPU ******")

    print("\n", "****** exp config ******")
    pprint(config)
    print("*************************\n")
    return config
    
def setRandomSeed(seed):
    # one random_seed in config file
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    print("****** set random seed as", seed, " ******\n")
    return seed

