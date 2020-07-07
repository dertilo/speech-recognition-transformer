import os
from os.path import join
from scripts.folder_tool import search_folder_for_post_fix_file_list, \
    extract_corpus_from_target_dict, merge
from tqdm import tqdm

def extract_target(file_list):
    name2target = {}
    for file in tqdm(file_list, desc='filelist'):
        with open(file, encoding='utf8') as reader:
            data = reader.readlines()
            for line in data:
                name = line.split(' ')[0]
                target = ' '.join(line.strip().split(' ')[1:]).lower()
                name2target[name] = target
    return name2target



def extract_name_fn(path):
    return path.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    HOME = os.environ['HOME']
    extracted_root = HOME+'/data/asr_data/ENGLISH/LibriSpeech'
    manifest_root = extracted_root + '/lightning_manifests'
    os.makedirs(manifest_root,exist_ok=True)
    corpus_root = extracted_root + '/lightning_corpus'
    os.makedirs(corpus_root,exist_ok=True)

    folders = ['dev-clean', 'dev-other', "train-clean-100", "train-clean-360",
              "train-other-500"]
    for folder in ['train-other-500']:
        extracted_to = join(extracted_root, folder)
        manifest_csv_path = join(manifest_root, folder + '.csv')
        corpus_path = join(corpus_root, folder + '.txt')

        #extract_nested_file(raw, extracted_to, 'tar')
        wav_list = search_folder_for_post_fix_file_list(extracted_to, '.flac')
        txt_list = search_folder_for_post_fix_file_list(extracted_to, '.txt')
        target_dict = extract_target(txt_list)
        extract_corpus_from_target_dict(target_dict, corpus_path)
        merge(wav_list, target_dict, extract_name_fn, manifest_csv_path)

    print('all done')