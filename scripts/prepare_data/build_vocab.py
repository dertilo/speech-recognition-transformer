import shutil
import sys
import os
sys.path.append(os.getcwd())
import sentencepiece as spm

if __name__ == '__main__':
    data_path = os.path.join(os.environ['HOME'], 'data/asr_data/ENGLISH/LibriSpeech/lightning_corpus')

    english_corpus = ['dev-clean.txt', 'dev-other.txt']
    for i in english_corpus:
        with open(os.path.join(data_path,i)) as reader, open('english.corpus', 'w') as writer:
            for i in reader:
                writer.write(i.strip() + '\n')

    model_name_prefix = 'librispeech'
    config_string = '--split_by_whitespace=1 --normalization_rule_name=nmt_nfkc_cf --add_dummy_prefix=1 --model_type=bpe --input=english.corpus --model_prefix={model_name} --vocab_size=3700 --pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3'.format(model_name=model_name_prefix)
    config_string += ' --user_defined_symbols=[S],[B],[N],[T],[P],[FIL],[SPK],'
    vocab_trainer = spm.SentencePieceTrainer.Train(config_string)

    shutil.move(model_name_prefix+'.model',os.path.join(data_path,model_name_prefix+'.model'))
    shutil.move(model_name_prefix+'.vocab',os.path.join(data_path,model_name_prefix+'.vocab'))


