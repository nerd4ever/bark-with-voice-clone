# From https://github.com/gitmylo/bark-voice-cloning-HuBERT-quantizer

import os.path
import shutil
import urllib.request
from typing import Optional

import huggingface_hub


class HuBERTManager:

    @staticmethod
    def get_dir(install_dir: Optional[str] = None):
        if install_dir is None:
            return os.path.join('data', 'models', 'hubert')
        return os.path.join(install_dir, 'data', 'models', 'hubert')

    @staticmethod
    def make_sure_hubert_installed(download_url: str = 'https://dl.fbaipublicfiles.com/hubert/hubert_base_ls960.pt',
                                   file_name: str = 'hubert.pt',
                                   base_dir: Optional[str] = None):
        install_dir = HuBERTManager.get_dir(base_dir)
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, file_name)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT base model')
            urllib.request.urlretrieve(download_url, install_file)
            print('Downloaded HuBERT')
        return install_file

    @staticmethod
    def make_sure_tokenizer_installed(model: str = 'quantifier_hubert_base_ls960_14.pth',
                                      repo: str = 'GitMylo/bark-voice-cloning', local_file: str = 'tokenizer.pth',
                                      base_dir: Optional[str] = None):
        install_dir = HuBERTManager.get_dir(base_dir)
        if not os.path.isdir(install_dir):
            os.makedirs(install_dir, exist_ok=True)
        install_file = os.path.join(install_dir, local_file)
        if not os.path.isfile(install_file):
            print('Downloading HuBERT custom tokenizer')
            huggingface_hub.hf_hub_download(repo, model, local_dir=install_dir, local_dir_use_symlinks=False)
            shutil.move(os.path.join(install_dir, model), install_file)
            print('Downloaded tokenizer')
        return install_file
