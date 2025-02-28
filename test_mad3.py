from tqdm import tqdm
from collections import defaultdict, deque
import numpy as np
import functools

from torch.utils.data import DataLoader
import torch
from config.all_config import AllConfig
from modules.baseline_pooling import BaselinePooling
from modules.metrics import sim_matrix_training, sim_matrix_inference, generate_embeds_per_video_id
from datasets.mad_dataset import MADDataset, collate_fn_replace_corrupted
from modules.metrics import t2v_metrics, v2t_metrics

text_embed_arr = []
vid_embed_arr = []
all_vid_ids = []
device = 'cuda'
pooling_type = 'topk'

mad_dataset = MADDataset(data_ratio=0.01)
collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=mad_dataset)
mad_data_loader = DataLoader(mad_dataset, batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

config = AllConfig()
pool_frames = BaselinePooling(pooling_type=pooling_type, config=config)
window_metric = defaultdict(lambda: deque())

for _, batch in tqdm(enumerate(mad_data_loader)):
    target, data, qid, windows = batch

    text_embed = data[0]['src_txt']
    vid_embed = data[0]['src_vid']

    video_features = vid_embed
    text_features = text_embed
    text_features = text_features[:, -1, :]

    video_features_pooled = pool_frames(text_features, video_features)

    sims = sim_matrix_training(text_features, video_features_pooled, pooling_type=pooling_type)
    sims = sims.unsqueeze(dim=1)

    metrics = t2v_metrics
    res = metrics(sims, target[0]['is_foreground'])

    # Compute window metrics
    for m in res:
        window_metric[m].append(res[m])

# Compute average of window metrics
for m in window_metric:
    res[m + "-window"] = np.mean(window_metric[m])

print(f"---------------------------------------------\n",
      f"R@1: {res['R1']} (window: {res['R1-window']})\n",
      f"R@5: {res['R5']} (window: {res['R5-window']})\n",
      f"R@10: {res['R10']} (window: {res['R10-window']})\n",
      #f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
      #f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
      )
