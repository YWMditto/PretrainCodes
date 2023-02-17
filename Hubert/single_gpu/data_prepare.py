import sys
import logging
from typing import Union, List, Any, Optional, Callable
from pathlib import Path
import soundfile as sf

import torch
from torch.utils.data import Dataset, Sampler, BatchSampler
import torch.nn.functional as F


logger = logging.getLogger(__name__)



def load_audio(manifest_path, max_keep_sample_size, min_keep_sample_size):
    if max_keep_sample_size is not None and min_keep_sample_size is not None:
        assert 0 <= min_keep_sample_size < max_keep_sample_size

    path_list = []
    size_list = []

    longer_num = 0
    shorter_num = 0
    with open(manifest_path, 'r') as f:
        root_path = Path(f.readline().rstrip())
        for line in f:
            sub_path, size = line.rstrip().split('\t')
            if max_keep_sample_size is not None and (size := int(size)) > max_keep_sample_size:
                longer_num += 1
            elif min_keep_sample_size is not None and size < min_keep_sample_size:
                shorter_num += 1
            else:
                path_list.append(str(root_path.joinpath(sub_path)))
                size_list.append(size)
    
    logger.info(
        (
            f"max_keep={max_keep_sample_size}, min_keep={min_keep_sample_size}, "
            f"loaded {len(path_list)}, skipped {shorter_num} short and {longer_num} long, "
            f"longest-loaded={max(size_list)}, shortest-loaded={min(size_list)}"
        )
    )

    return path_list, size_list


def load_label(label_path, label_processor=None):
    label_list = []
    with open(label_path, 'r') as f:
        for line in f:
            line = line.rstrip().split()
            if label_processor is not None:
                line = label_processor(line)
            label_list.append(line)
    return label_list
    

def verify_label_lengths(
    audio_size_list,
    audio_path_list,
    sample_rate,
    label_list,
    label_rate,
    tol=0.1
):
    # 根据 sample rate 和 label rate 检测音频和其对应的标签在时间长度上是否一致；
    assert label_rate > 0

    label_size_list = [len(w) for w in label_list]
    assert len(audio_size_list) == len(label_size_list), f"audio 数量和 label 数量不一致：{len(audio_size_list)}, {len(label_size_list)}."

    num_invalid = 0
    for idx, audio_size, label_size in enumerate(zip(audio_size_list, label_size_list)):
        if ((audio_seconds := audio_size / sample_rate) - (label_seconds := label_size / label_rate)) > tol:
            logger.warning(
                (
                    f"audio and label duration differ too much "
                    f"(|{audio_seconds} - {label_seconds}| > {tol}) "
                    f"of audio {audio_path_list[idx]}. Check if `label_rate` "
                    f"is correctly set (currently {label_rate}). "
                    f"num. of samples = {audio_size}; "
                    f"label length = {label_size}."
                )
            )
            num_invalid += 1

    if num_invalid > 0:
        logger.warning(
            f"total {num_invalid} (audio, label) pairs with mismatched lengths"
        )



class HubertPretrainDataset(Dataset):
    """
    用于 Hubert 预训练下的数据集；

    需要注意的地方：
        1. 加载预处理好的 wav manifest 文件 (.tsv) 和伪标签文件；
            a. 去除过长以及过短的音频；
            b. 检查每条音频长度和其按照 frame rate 转换得到的标签长度是否一致；
        2. 音频在具体拿时才会加载，需要考虑音频是否 normalize；fairseq 会习惯性地为 dataset 传入一个 label_processors 实际上就是一个字典来编码标签，这里我们不需要；
        3. 在 collate 时需要设置是否是 pad 还是 crop；pad 就是将一个 batch 中的所有音频全部 pad 到一个长度（设置的最长长度或者batch中的最长长度），
         crop 则是将所有音频全部裁剪到最短长度，因此这里需要的一个额外的操作就是需要按照时间片段对标签进行裁剪，这个应该由 collate fn 完成；
        4. order indices，按照音频长度对数据进行排序，fairseq 是因为整体代码结构的原因直接写在了数据集当中，实际上这个工作应该由 batch sampler 完成，

    *** fairseq 因为框架的原因，将大量的操作（包括 filter by size, order indices, distributed sampler）全部集中在 dataset、task.get_batch_iterator 以及 trainer.get_train_iterator 中，并且自己设计了整体的 iterator，这里
    我们仍旧按照传统 pytorch 的结构，通过 sampler, batch sampler 和 collate fn 的方式来进行编写，尽量以最简单的实现来满足实际在训练时使用到的功能；
    """

    def __init__(
        self,
        manifest_path: str, # 音频的 tsv 文件；
        sample_rate: float, # 模型实际需要的 sample rate；
        label_path: str, # 标签对应的 km 文件；
        label_rate: float, # 标签对应的时间压缩率，例如 50 就表示原本 1s 的音频对应 50 个标签；
        max_keep_sample_size: Optional[int] = None, # 和 min 一起用于将不符合长度的音频筛选掉；
        min_keep_sample_size: Optional[int] = None, 
        normalize: bool = False, # 是否对音频进行正则化；
        label_processor: Optional[Callable] = None # 用于对读入的标签进行额外的处理，例如可能需要加入特殊的 token（pad、mask等）从而使得标签的idx后移；
    ):

        self.audio_path_list, self.audio_size_list = load_audio(
            manifest_path=manifest_path,
            max_keep_sample_size=max_keep_sample_size,
            min_keep_sample_size=min_keep_sample_size
        )        
        self.label_list = load_label(label_path=label_path, label_processor=label_processor)
        verify_label_lengths(
            audio_size_list=self.audio_size_list,
            audio_path_list=self.audio_path_list,
            sample_rate=sample_rate,
            label_list=self.label_list,
            label_rate=label_rate
        )

        self.manifest_path = manifest_path
        self.sample_rate = sample_rate
        self.label_path = label_path
        self.label_rate = label_rate
        self.max_keep_sample_size = max_keep_sample_size
        self.min_keep_sample_size = min_keep_sample_size
        self.normalize = normalize
        self.label_processor = label_processor

    def get_audio(self, index):
        audio_path = self.audio_path_list[index]
        audio, cur_sample_rate = sf.read(audio_path)
        audio = torch.from_numpy(audio).float()
        audio = self.postprocess(audio, cur_sample_rate)
        return audio

    def postprocess(self, wav, cur_sample_rate):
        if wav.dim() == 2:
            wav = wav.mean(-1)
        assert wav.dim() == 1, wav.dim()

        if cur_sample_rate != self.sample_rate:
            raise Exception(f"sr {cur_sample_rate} != {self.sample_rate}")

        if self.normalize:
            with torch.no_grad():
                wav = F.layer_norm(wav, wav.shape)
        return wav

    def get_label(self, index):
        return self.label_list[index]
    
    def __len__(self) -> int:
        return len(self.audio_path_list)
    
    def __getitem__(self, index):
        audio = self.get_audio(index)
        label = self.get_label(index)
        return {"idx": index, "audio": audio, "label": label}
    
    def num_tokens(self, index) -> int:
        # 用来 order indices；
        return self.audio_size_list[index]

    def size_list(self):
        # 用来 order indices，collate；
        return self.audio_size_list



"""
fairseq 中对于数据的实际拿取的过程：

    1. filter by size，如果设置，则对数据集再次进行筛选；
    2. order indices 或者 bucket order indices，对数据按照长度进行排序或者分成几个 ``桶``，每个桶中按照长度进行排序；
    3. collate：
        a. constant token batch，一个 batch 中数据长度尽可能一样，并且按照加起来的总长度决定一个 batch 的大小；
        b. pad or crop，按照设置进行 pad 或者 crop，注意同时需要对 label 进行 pad 和 crop；

注意这里的操作的顺序是和 fairseq 中的顺序保持一致的；
"""



class HubertBatchSampler:
    """
    这里将原本就属于 batch sampler 的操作从 dataset 中移到该类中；


    *** 该 batch sampler 在 multi gpu 中需要进行重写；
    """

    



















