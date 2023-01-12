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
device = 'cuda'
pooling_type = 'topk'
k=50

mad_dataset = MADDataset(data_ratio=0.01)
collate_fn = functools.partial(collate_fn_replace_corrupted, dataset=mad_dataset)
mad_data_loader = DataLoader(mad_dataset, batch_size=1,
                             shuffle=False, num_workers=0,
                             collate_fn=collate_fn)

config = AllConfig()
config.k = k
pool_frames = BaselinePooling(pooling_type=pooling_type, config=config)
window_metric = defaultdict(lambda: deque())

for _, batch in tqdm(enumerate(mad_data_loader)):
    all_vid_ids = []

    target, data, qid, windows = batch

    for data_idx in range(len(data)):
        for _ in range(data[data_idx]['src_vid'].shape[0]):
            all_vid_ids.append(qid[data_idx])

    # num_vids x frames x embed_dim
    vid_embeds = torch.stack([d['src_vid'][0] for d in data])
    text_embeds = torch.vstack([d['src_txt'][:,0,:] for d in data])

    # Pool frames for inference once we have all texts and videos
    pool_frames.cpu()

    vid_embeds_pooled = pool_frames(text_embeds, vid_embeds)  # (movies,bsz*movies,embed)

    pool_frames.cuda()

    # num_vids x max_text_per_vid x embed_dim, (num_vids x num_vids x max_text_per_vid x embed_dim)
    text_embeds_per_video_id, vid_embeds_pooled_per_video_id = generate_embeds_per_video_id(text_embeds,
                                                                                            vid_embeds_pooled, all_vid_ids,
                                                                                            pooling_type)

    # num_vids x max_text_per_vid x num_vids
    sims = sim_matrix_inference(text_embeds_per_video_id, vid_embeds_pooled_per_video_id, pooling_type)

    metrics = t2v_metrics
    res = metrics(sims)

    # Compute window metrics
    for m in res:
        window_metric[m].append(res[m])

# Compute average of window metrics
for m in window_metric:
    res[m + "-window"] = np.mean(window_metric[m])

print(f"-------------------------------------------\n",
      f"R@1: {res['R1']} (window: {res['R1-window']})\n",
      f"R@5: {res['R5']} (window: {res['R5-window']})\n",
      f"R@10: {res['R10']} (window: {res['R10-window']})\n",
      f"MedR: {res['MedR']} (window: {res['MedR-window']})\n",
      f"MeanR: {res['MeanR']} (window: {res['MeanR-window']})\n")

