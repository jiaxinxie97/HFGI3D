from random import choice
from string import ascii_uppercase
import sys 
sys.path.append("..")
sys.path.append("../..")
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import os

import shutil
from configs import global_config
import wandb

from training.coaches.multi_id_coach import MultiIDCoach
from training.coaches.single_id_coach import SingleIDCoach
from utils.ImagesDataset import ImagesDataset


def run_PTI(path_config,run_name='',multi_views=False,use_wandb=False, use_multi_id_training=False):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = global_config.cuda_visible_devices

    if run_name == '':
        global_config.run_name = ''.join(choice(ascii_uppercase) for i in range(12))
    else:
        global_config.run_name = run_name

    if use_wandb:
        run = wandb.init(project=paths_config.pti_results_keyword, reinit=True, name=global_config.run_name)
    global_config.pivotal_training_steps = 1
    global_config.training_step = 1

    embedding_dir_path = f'{paths_config.embedding_base_dir}/{paths_config.input_data_id}/{paths_config.pti_results_keyword}'
    os.makedirs(embedding_dir_path, exist_ok=True)

    dataset = ImagesDataset(paths_config.input_data_path, paths_config.name, transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])]))

    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    if use_multi_id_training:
        coach = MultiIDCoach(dataloader, paths_config,multi_views,use_wandb)
    else:
        coach = SingleIDCoach(dataloader,paths_config,multi_views, use_wandb)

    coach.train()

    return global_config.run_name


if __name__ == '__main__':

    example_config = sys.argv[1]
    print(example_config)
    shutil.copy(example_config,'../../example_configs/config.py')
    import example_configs.config as paths_config
    run_PTI(paths_config,run_name='',multi_views=False,use_wandb=False, use_multi_id_training=False)
    os.system('python ../../multi_views/generate_multi_views.py  %s %s %s %s'%(paths_config.name,paths_config.experiments_output_dir,paths_config.input_data_path,paths_config.multi_views_output_dir))
    if paths_config.optimization:
        run_PTI(paths_config,run_name='',multi_views=True,use_wandb=False, use_multi_id_training=False)
    else:
        if os.path.exists(f'{paths_config.experiments_output_dir}/{paths_config.name}'):
            shutil.rmtree(f'{paths_config.experiments_output_dir}/{paths_config.name}')
        shutil.copytree(f'{paths_config.multi_views_output_dir}/{paths_config.name}',f'{paths_config.experiments_output_dir}/{paths_config.name}')
        shutil.move(f'{paths_config.experiments_output_dir}/{paths_config.name}/{paths_config.name}.png',f'{paths_config.experiments_output_dir}/{paths_config.name}.png')
        shutil.move(f'{paths_config.experiments_output_dir}/{paths_config.name}/{paths_config.name}.mp4',f'{paths_config.experiments_output_dir}/{paths_config.name}.mp4')