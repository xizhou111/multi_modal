import json
import torch
from tqdm import tqdm
from multiprocessing import Pool
from util.milvus_tools import MyMilvus

import sys
sys.path.append('..')
from blip2_models.blip2_qformer import Blip2Qformer
from dataset import MultiModelDataset
from torch.utils.data import DataLoader

from safetensors.torch import load_file

import argparse



def generate_tiku_features(jmjx_tiku_data_path, model_path, device):
    with open(jmjx_tiku_data_path, 'r') as f:
        jmjx_tiku_data = json.load(f)

    model = Blip2Qformer()
    model.to('cuda:{}'.format(device))
    model.load_state_dict(load_file(model_path), strict=False)
    model.eval()

    tiku_dataset = MultiModelDataset(jmjx_tiku_data)
    tiku_loder = DataLoader(tiku_dataset, batch_size=64, shuffle=False, num_workers=16)  # 减小批处理大小
    with torch.no_grad():
        for batch in tqdm(tiku_loder, total=len(tiku_loder), desc='Extracting features'):
            _id_list = batch['image_id']
            question_text_list = batch['text_input']
            text_features_list = model.extract_features(batch, mode='text')['text_embeds_proj'][:,0,:].tolist()

            for _id, question_text, text_features in zip(_id_list, question_text_list, text_features_list):
                tiku_feature = {
                    'id': _id,
                    'question_text': question_text,
                    'text_features': text_features
                }

                with open('/mnt/cfs/NLP/zcl/multi_modal/eval/tiku_features_data/jmjx_tiku_features_part{}.jsonl'.format(device), 'a') as f:  # 在处理完每个特征后，立即将其写入文件
                    f.write(json.dumps(tiku_feature, ensure_ascii=False) + '\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--part', type=int, required=True, help='Part number')
    args = parser.parse_args()

    jmjx_tiku_data_path = '/mnt/cfs/NLP/zcl/multi_modal/data/new/jmjx_tiku_cleaned/jmxj_tiku_data_part{}.json'.format(args.part)
    print(jmjx_tiku_data_path)
    model_path = '/mnt/cfs/NLP/zcl/multi_modal/output_eva_clip_g_chinese-roberta-wwm-ext_448_i2t_0811_baiduyun2/checkpoint-50000' + '/model.safetensors'
    generate_tiku_features(jmjx_tiku_data_path, model_path, device=args.part)
