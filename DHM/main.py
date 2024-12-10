# Code is heavily borrowed from https://github.com/vcg-uvic/learned-correspondence-release
# Author: Jiahui Zhang
# Date: 2019/09/03
# E-mail: jiahui-z15@mails.tsinghua.edu.cn
import os
#os.environ["CUDA_VISIBLE_DEVICES"] = "7"
os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'

from config import get_config, print_usage
config, unparsed = get_config()
import os
os.environ['CUDA_VISIBLE_DEVICES'] = config.gpu_id
#os.environ["HDF5_USE_FILE_LOCKING"] = 'FALSE'
#os.environ["KMP_DUPLICATE_LIB_OK"]='TRUE'
import torch.utils.data
import sys
from data import collate_fn, CorrespondencesDataset
from DHM import DHMNet as Model
from train import train
from test import test


def create_log_dir(config):
    if not os.path.isdir(config.log_base):
        os.makedirs(config.log_base)
    if config.log_suffix == "":
        suffix = "-".join(sys.argv)
    result_path = config.log_base
    if not os.path.isdir(result_path):
        os.makedirs(result_path)
    if not os.path.isdir(result_path+'/train'):
        os.makedirs(result_path+'/train')
    if not os.path.isdir(result_path+'/valid'):
        os.makedirs(result_path+'/valid')
    if not os.path.isdir(result_path+'/test'):
        os.makedirs(result_path+'/test')
    if os.path.exists(result_path+'/config.th'):
        print('warning: will overwrite config file')
    torch.save(config, result_path+'/config.th')
    # path for saving traning logs
    config.log_path = result_path+'/train'

def main(config):
    """The main function."""
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Initialize network
    model = Model(config)
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0,2'
    # device_ids = [0,1]
    #model = torch.nn.DataParallel(model, device_ids=device_ids)
    
    #device = torch.device("cuda:{}".format("0,1") if torch.cuda.is_available() else "cpu")
    #model = torch.nn.DataParallel(model, device_ids = [0,1])

    #device=torch.device("cuda:1" )
    #model.to(device)
    #print('generated parameters:', sum(param.numel() for param in model.parameters()))
    # Run propper mode
    if config.run_mode == "train":
        create_log_dir(config)

        train_dataset = CorrespondencesDataset(config.data_tr, config)

        train_loader = torch.utils.data.DataLoader(
                train_dataset, batch_size=config.train_batch_size, shuffle=True,
                num_workers=16, pin_memory=False, collate_fn=collate_fn)

        valid_dataset = CorrespondencesDataset(config.data_va, config)
        valid_loader = torch.utils.data.DataLoader(
                valid_dataset, batch_size=config.train_batch_size, shuffle=False,
                num_workers=8, pin_memory=False, collate_fn=collate_fn)
        #valid_loader = None
        print('start training .....')
        train(model, train_loader, valid_loader, config)

    elif config.run_mode == "test":
        test_dataset = CorrespondencesDataset(config.data_te, config)
        test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=1, shuffle=False,
                num_workers=8, pin_memory=False, collate_fn=collate_fn)

        test(test_loader, model, config)




if __name__ == "__main__":


    config, unparsed = get_config()
    if len(unparsed) > 0:
        print_usage()
        exit(1)

    main(config)

