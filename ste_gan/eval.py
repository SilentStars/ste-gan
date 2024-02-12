import argparse
import logging
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf

import ste_gan
from ste_gan.constants import DataType
from ste_gan.data.emg_dataset import EMGDataset
from ste_gan.data.loader import loaders_via_config
from ste_gan.utils.plot_utils import plot_real_vs_fake_emg_signal_with_envelope
from ste_gan.losses.time_domain_loss import MultiTimeDomainFeatureLoss
from ste_gan.models.discriminator import init_emg_discriminators
from ste_gan.models.emg_encoder import load_emg_encoder
from ste_gan.models.generator import init_emg_generator
from ste_gan.train_utils import (add_eval_hyperparams_to_parser, load_config)
from ste_gan.utils.common import load_latest_checkpoint

from utils.common import fix_state_dict



def eval(cfg, eval_path, device, emg_enc_ckpt):
    device = torch.device(device)

    logging.info(f"Initializing Models")
   
    netG = init_emg_generator(cfg)
    netG.to(device)
    
    def load_trained_models(checkpoint_dir, generator):
        generator.load_state_dict(fix_state_dict(torch.load(checkpoint_dir + 'best_netG.pt', map_location=device)))
        return generator
        
    netG = load_trained_models(eval_path, netG)
    
    train_loader, valid_loader, test_loader = loaders_via_config(cfg)
    
    valid_data_set: EMGDataset = valid_loader.dataset
    
    speech_feature_type = cfg.model.speech_feature_type
    
    def generate_samples(save_fig_path):
        # save_start = time.time()
        netG.eval()
        for i, (sample_dict) in enumerate(valid_data_set):
            with torch.no_grad():
                s_t = sample_dict[speech_feature_type].unsqueeze(0).to(device)
                sess_idx = sample_dict[DataType.SESSION_INDEX].unsqueeze(0).to(device)
                spk_mode_idx = sample_dict[DataType.SPEAKING_MODE_INDEX].unsqueeze(0).to(device) 

                # Generate the fake EMG signal
                pred_emg = netG.generate(s_t, sess_idx, spk_mode_idx).squeeze(0).detach().cpu().numpy()

                # Generate the real EMG signal
                real_emg = sample_dict[DataType.REAL_EMG].squeeze(0).detach().cpu().numpy()
                
                plot_real_vs_fake_emg_signal_with_envelope(
                    real_emg_signal=real_emg,
                    fake_emg_signal=pred_emg,
                    file_id=f"Validation sample {i}",
                    save_as=save_fig_path + 'figs/emg_compare_sample_' + str(i) + '.png',
                    tb_summary_writer=None,
                    tb_tag_prefix="val/envelopes_emg_real_vs_fake",
                    show=False
                )
            if i > cfg.train.num_test_samples:
                break
    
    generate_samples(eval_path)

def main(cfg: DictConfig, continue_run: bool, debug: bool, emg_enc_ckpt: Path, **kwargs):
    dataset_root = cfg.data.dataset_root
    print(f"Data root: {dataset_root}")
    print(f"continue_run: {continue_run}")
    print(f"Debug (argparse): {debug}")
    
    if not debug and cfg.train.debug:
        print(f"WARNING: SETTING GLOBAL DEBUG FLAG")
        debug = True
    
    # Create output dir
    # model_base_dir = Path(cfg.model_base_dir)
    # output_directory = model_base_dir/ create_ste_gan_model_name(
    #     cfg, add_timestamp=False, debug=debug,
    # )
    # if output_directory.exists() and continue_run:
    #     logging.info(f"WARNING: Removing old model directory: {output_directory}")
    #     checkpoint = output_directory
    # else:
    #     checkpoint = None
    # output_directory.mkdir(exist_ok=True, parents=True)
    # print(f"Output directory: {output_directory}")

    # done_file = output_directory / ".done"
    # if output_directory and done_file.exists():
    #     logging.warning(f"Exiting training script as '.done' file exists: {done_file.absolute()}")
    #     sys.exit()

    # Save configuration
    # config_file = output_directory / "config.yaml"
    # if not config_file.exists():
    #     with open(config_file, '+w') as fp:
    #         OmegaConf.save(config=cfg, f=fp.name)

    # logging.info(OmegaConf.to_yaml(cfg))
    # logging.getLogger().setLevel(logging.INFO)
    # log_file = output_directory / "log.txt"
    # fh = logging.FileHandler(str(log_file.absolute()))
    # logging.getLogger().addHandler(fh) 

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device='cpu'
    eval(cfg, 'eval/', device, emg_enc_ckpt)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/ste_gan_base_gantts.yaml",
                        help="The main training configuration for this run.")
    parser.add_argument("--data", type=str, default="configs/data/gaddy_and_klein_corpus.yaml",
                        help="A path to a data configuration file.")
    parser.add_argument("--emg_enc_cfg", type=str, default="configs/emg_encoder/conv_transformer.yaml",
                        help="A path to an EMG encoder configuration file.")
    parser.add_argument("--emg_enc_ckpt", type=str, default="exp/emg_encoder/EMGEncoderTransformer_voiced_only__seq_len__200__data_gaddy_complete/best_val_loss_model.pt",
                        help="A path to a checkpooint of a pre-trained EMG encoder. Must correspond to the EMG encoder configuration in 'emg_enc_cfg'.")
    parser.add_argument(
        '--checkpoint',
        type=Path,
        help='Optional checkpoint to start training from')
    parser.add_argument(
        '--continue_run',
        action='store_true',
        help='Whether to continue training')
    parser.add_argument(
        '--debug',
        action='store_true',
        help='Whether to run the training script in debug mode.')
    
    parser = add_eval_hyperparams_to_parser(parser)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    cfg = load_config(args)
    args.cfg = cfg
    main(**vars(args))
    
    
    