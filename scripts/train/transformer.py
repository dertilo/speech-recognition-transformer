import os
import sys
sys.path.append(os.getcwd())

from pytorch_lightning import Trainer
import argparse
import random
import torch as t
import torch.backends.cudnn as cudnn
from warnings import filterwarnings
from pytorch_lightning.logging import TestTubeLogger
from pytorch_lightning.callbacks import ModelCheckpoint

from src.model.transformer.lightning_model import LightningModel


def get_args():
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('--exp_root', 'exp', default=str)
    parent_parser.add_argument('--log_root', 'lightning_logs', default=str)
    parent_parser.add_argument('-seed', default=1)
    parent_parser.add_argument('-epochs', default=1000)
    parser = LightningModel.add_model_specific_args(parent_parser)
    return parser.parse_args()



def main(hparams):
    model = LightningModel(hparams)
    if hparams.seed is not None:
        random.seed(hparams.seed)
        t.manual_seed(hparams.seed)
        cudnn.deterministic = True
    exp_root = hparams.exp_root
    log_folder = hparams.log_folder
    log_root = os.path.join(exp_root, log_folder)
    logger = TestTubeLogger(exp_root, name=log_folder, version=None)
    # checkpoint = ModelCheckpoint(filepath=)
    # logger = TestTubeLogger(exp_root, name=log_folder, version=214)
    # checkpoint = ModelCheckpoint(filepath='exp/lightning_logs/version_214/checkpoints/',
    #                              monitor='wer', verbose=1, save_best_only=False)
    trainer = Trainer(
        # logger=logger,
        # checkpoint_callback=checkpoint,
        # fast_dev_run=True,
        # overfit_pct=0.03,
        default_save_path='exp/',
        val_check_interval=1.0,
        log_save_interval=50000,
        row_log_interval=50000,
        gpus=1,
        nb_gpu_nodes=hparams.nb_gpu_nodes,
        max_nb_epochs=hparams.epochs,
        gradient_clip_val=5.0,
        min_nb_epochs=3000,
        use_amp=True,
        amp_level='O2',
        nb_sanity_val_steps=0
    )
    # if hparams.evaluate:
    #     trainer.run_evaluation()
    # else:
    trainer.fit(model)

if __name__ == '__main__':
    main(get_args())