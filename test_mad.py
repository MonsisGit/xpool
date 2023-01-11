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
eval_window_size = 5
num_frames = 5

mad_dataset = MADDataset(data_ratio=0.001)
collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=mad_dataset)
mad_data_loader = DataLoader(mad_dataset, batch_size=1,
                             shuffle=True, num_workers=0,
                             collate_fn=collate_fn)

config = AllConfig()
pool_frames = BaselinePooling(pooling_type='avg', config=config)

for _, batch in tqdm(enumerate(mad_data_loader)):

    target, data, qid, windows = batch
    text_embed = data[0]['src_txt']
    vid_embed = data[0]['src_vid']

    text_embed_arr.append(text_embed.cpu())
    vid_embed_arr.append(vid_embed.cpu())

    text_features = text_embed / text_embed.norm(dim=-1, keepdim=True)
    video_features = vid_embed / vid_embed.norm(dim=-1, keepdim=True)
    text_features = text_features[:, 0, :]

    video_features_pooled = pool_frames(text_features, video_features)

    sims = sim_matrix_training(text_features, video_features_pooled, pooling_type='avg')
    sims = sims.unsqueeze(dim=1)

    metrics = t2v_metrics
    res = metrics(sims)

    window_metric = defaultdict(lambda: deque(maxlen=eval_window_size))
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
      f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
      f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n",
      )
