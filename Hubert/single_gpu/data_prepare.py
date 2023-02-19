import sys
import logging
from typing import Union, List, Any, Optional, Callable
from pathlib import Path
import soundfile as sf
import numpy as np
from itertools import chain

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
    3. constant token batch，一个 batch 中数据长度尽可能一样，并且按照加起来的总长度决定一个 batch 的大小；
    4. collate: pad or crop，按照设置进行 pad 或者 crop，注意同时需要对 label 进行 pad 和 crop；

注意这里的操作的顺序是和 fairseq 中的顺序保持一致的；
此外 fairseq 的数据流程，重写各种 iterator 以及具体的数据拿取过程（例如预取），可能进行了大量的优化，这里因为我们主要是想评测一下 deepspeed
对于模型训练的优化，因此按照最简单的 pytorch 的基本写法来编写；
"""


class HubertBatchSampler:
    """
    这里将原本就属于 batch sampler 的操作从 dataset 中移到该类中；

    该类主要负责：
        1. filter by size，如果设置，则对数据集再次进行筛选；
        2. order indices 或者 bucket order indices，对数据按照长度进行排序或者分成几个 ``桶``，每个桶中按照长度进行排序；
        3. constant token batch，一个 batch 中数据长度尽可能一样，并且按照加起来的总长度决定一个 batch 的大小；
            这一步整体的思想就是先按照长度排序，然后分桶；
            如果随机，那么需要 a. 打乱桶之间的顺序；b. 打乱桶内数据的顺序；（如果不随机，那么实际上分桶也没有意义；）
            然后每次 iter 时，将所有桶的数据连在一起，然后按照一个 batch 的 token 数量打包成一个个 batch；

    *** 该 batch sampler 在 multi gpu 中需要进行重写，不过并不复杂，主要的操作在于按照进程数量对数据集进行切片；

    该类用于替换 pytorch dataloader 中原本的 BatchSampler，对于一个 batch sampler 来说，需要考虑以下问题：
        1. 该类是一个可迭代对象，需要实现 __iter__ 方法；
        2. 需要考虑在遍历过程中再次调用 iter 方法是否需要重置该类的内部状态，然后返回一个全新的循环；（通常来讲每次调用 iter 都返回一个全新的循环；）
        3. 需要考虑每次 iter 时是否再对数据的遍历顺序进行随机地打乱；
    """
    def __init__(
        self,
        size_list: List[int],
        one_batch_total_tokens: int,
        filter_size: Optional[int] = None,
        num_buckets: Optional[int] = None,
        shuffle: bool = False,
        seed: int = 0,
        dataset_name: str = 'tmp',
        **kwargs
    ):
        logger.info(f"Use ``HubertBatchSampler``, current dataset is {dataset_name}.")

        self.size_list = size_list
        self.one_batch_total_tokens = one_batch_total_tokens
        self.filter_size = filter_size
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self._seed = seed
        self.dataset_name = dataset_name
        self.epoch = kwargs.get('epoch', 0)

        self.size_list_with_indices = [(idx, w) for idx, w in enumerate(size_list)]
        self.size_list_with_indices = sorted(self.size_list_with_indices, key=lambda x: x[1])

        if filter_size is not None:
            filter_size = min(filter_size, one_batch_total_tokens)
        else:
            filter_size = one_batch_total_tokens

        self.size_list_with_indices, longer_num = self.filter_by_size(self.size_list_with_indices, filter_size, has_sorted=True)
        logger.warning(f"Dataset {dataset_name} has {longer_num} samples whose lengths are longer than "
                       f"one_batch_total_tokens: {one_batch_total_tokens} or filter_size: {filter_size}.")

        # 提前分桶；
        if self.shuffle and num_buckets is not None and num_buckets > 1:
            each_bucket_num = len(self.size_list_with_indices) // num_buckets
            indices_buckets = []
            for i in range(num_buckets):
                indices_buckets.append(self.size_list_with_indices[i*each_bucket_num: (i+1)*each_bucket_num])
            indices_buckets[-1].extend(self.size_list_with_indices[(i+1)*each_bucket_num: ])
        else:
            indices_buckets = [self.size_list_with_indices]
        self.indices_buckets = indices_buckets
        self.batches = self.batchify()

    @staticmethod
    def filter_by_size(size_list_with_indices, filter_size, has_sorted=False):
        tmp_list = []
        longer_num = 0
        if not has_sorted:
            for sample in size_list_with_indices:
                if sample[1] <= filter_size:
                    tmp_list.append(sample)
                else:
                    longer_num += 1
        else:
            for ii in range(len(size_list_with_indices)-1, -1, -1):
                if size_list_with_indices[ii][1] <= filter_size:
                    break
                else:
                    longer_num += 1
            tmp_list = size_list_with_indices[:ii+1]
        return tmp_list, longer_num

    def set_epoch(self, epoch):
        self.epoch = epoch

    @property
    def seed(self) -> int:
        return abs(self._seed + self.epoch)

    def batchify(self):
        if self.shuffle:
            rng = np.random.default_rng(seed=self.seed)
            rng.shuffle(self.indices_buckets)
            for bucket in self.indices_buckets:
                rng.shuffle(bucket)

        batches = []
        cur_batch_max_length = -1
        cur_idx = 0
        cur_batch = []
        indices = list(chain(*self.indices_buckets))

        while cur_idx < len(indices):
            sample_idx, sample_length = indices[cur_idx]
            max_length = max(cur_batch_max_length, sample_length)
            if max_length * (len(cur_batch) + 1) <= self.one_batch_total_tokens:
                cur_batch.append(sample_idx)
            else:
                batches.append(cur_batch)
                cur_batch = [sample_idx]
            cur_batch_max_length = max_length
            cur_idx += 1
        batches.append(cur_batch)

        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(batches)

        # 将 token 数量最多的 batch、单个数据长度最长的 batch 以及 batch size 最大的 batch 放在最前面，从而提前发现 OOM；
        max_tokens_batch_idx = np.argmax([sum(self.size_list[w] for w in _batch) for _batch in batches])
        max_length_batch_idx = np.argmax([max(self.size_list[w] for w in _batch) for _batch in batches])
        max_size_batch_idx = np.argmax(map(len, batches))
        for idx in {max_tokens_batch_idx, max_length_batch_idx, max_size_batch_idx}:
            batch = batches.pop(idx)
            batches.insert(0, batch)

        logger.info(f"Dataset {self.dataset_name} uses total {len(indices)} samples and is packed into {len(batches)} batches.")
        return batches

    def __iter__(self):
        for batch in self.batches:
            yield batch

        if self.shuffle:
            self.batches = self.batchify()

    def __len__(self):
        return len(self.batches)


class HubertCollater:
    def __init__(
        self,
        max_sample_size: int,
        pad_audio: bool,
        random_crop: bool,
        pad_token_id: int,
        sample_rate: int = 16000,
        label_rate: int = 50
    ):
        self.max_sample_size = max_sample_size
        self.pad_audio = pad_audio
        self.random_crop = random_crop
        self.pad_token_id = pad_token_id
        self.sample_rate = sample_rate
        self.label_rate = sample_rate

        logger.info(f"HubertCollater is configured as: \n"
                    f"\tmax_sample_size: {max_sample_size},"
                    f"\tpad_audio: {pad_audio},"
                    f"\trandom_crop: {random_crop},"
                    f"\tpad_token_id: {pad_token_id},"
                    f"\tsample_rate: {sample_rate},"
                    f"\tlabel_rate: {label_rate}.")

        self.s2f = label_rate / sample_rate

    def collate_fn(self, batch):
        """
        hubert 的预训练默认是将一个 batch 中的所有音频全部随机裁剪到最短长度；

        :param batch:
        :param max_sample_size: 将全部音频的最大长度限制到这个数；
        :param pad_audio: 是否对一个 batch 中的音频全部 pad 到它们中的最长的长度或者 max_sample_size；如果为 False，那么就将所有音频裁剪
         到这个 batch 中的最短的长度；
        :param random_crop: 如果不 pad 到最长，那么是否在音频中随机裁剪一段，还是说全部默认从 0 开始裁剪到对应长度；
        :param pad_token_id: 用于 pad 标签序列，这里需要和 label_processors 保持一致；
        :return:
        """
        sample_list = [s for s in batch if s["audio"] is not None]
        if len(sample_list) == 0:
            return {}

        audio_list = [s["audio"] for s in sample_list]
        audio_size_list = [len(s) for s in audio_list]
        label_list = [s['label'] for s in sample_list]

        if self.pad_audio:
            # collate audio;
            audio_size = min(max(audio_size_list), self.max_sample_size)
            collated_audios = audio_list[0].new_zeros(len(audio_list), audio_size)
            # wav2vec2 中预训练 base 和 large 都没有使用 padding mask（指加载数据时，但实际上 fairseq 使用了 require len multiple of），而在微调时使用；
            #  fairseq 实现的 hubert 本身则全部默认使用 padding mask；
            # 这里我们按照 transformers 的设定，1 表示需要 attend，0 表示不需要，因此 0 表示该位置是 pad；需要在实现模型的时候考虑到这一点；
            padding_mask = torch.BoolTensor(collated_audios.shape).fill_(True)
            audio_start_list = [0 for _ in range(len(collated_audios))]
            for i, audio in enumerate(audio_list):
                diff = len(audio) - audio_size
                if diff == 0:
                    collated_audios[i] = audio
                elif diff < 0:
                    collated_audios[i] = torch.cat([audio, audio.new_full((-diff,), 0.0)])
                    padding_mask[i, diff:] = False
                else:
                    raise RuntimeError

            # collate label;
            frame_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            frame_size = int(round(audio_size * self.s2f))
            label_list = [torch.LongTensor(t[s: s + frame_size]) for t, s in zip(label_list, frame_start_list)]
            lengths = torch.LongTensor([len(t) for t in label_list])
            ntokens = lengths.sum().items()
            collated_labels = label_list.new((len(label_list), frame_size)).fill_(self.pad_token_id)
            for i, label in enumerate(label_list):
                collated_labels[i, :len(label)] = label
        else:
            # collate audio;
            audio_size = min(min(audio_size_list), self.max_sample_size)
            collated_audios = audio_list[0].new_zeros(len(audio_list), audio_size)
            padding_mask = torch.BoolTensor(collated_audios.shape).fill_(True)
            audio_start_list = [0 for _ in range(len(collated_audios))]
            for i, audio in enumerate(audio_list):
                diff = len(audio) - audio_size
                if diff == 0:
                    collated_audios[i] = audio
                elif diff > 0:
                    start, end = 0, audio_size
                    if self.random_crop:
                        start = np.random.randint(0, diff + 1)
                        end = start + audio_size
                    collated_audios[i] = audio[start: end]
                    audio_start_list[i] = start
                else:
                    raise RuntimeError
            # collate label;
            frame_start_list = [int(round(s * self.s2f)) for s in audio_start_list]
            frame_size = int(round(audio_size * self.s2f))
            rem_size_list = [len(t) - s for t, s in zip(label_list, frame_start_list)]
            frame_size = min(frame_size, *rem_size_list)
            label_list = [torch.LongTensor(t[s: s + frame_size]) for t, s in zip(label_list, frame_start_list)]
            lengths = torch.LongTensor([len(t) for t in label_list])
            ntokens = lengths.sum().items()
            collated_labels = torch.cat(label_list)

        collated_batch = {
            "idx": torch.LongTensor([s['idx'] for s in batch]),
            "net_input": {
                "audio": collated_audios,
                "padding_mask": padding_mask,
            },
            "label_lengths": lengths,
            "label_ntokens": ntokens,
            "labels": collated_labels
        }

        return collated_batch



















