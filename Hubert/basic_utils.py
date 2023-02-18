import random
from pathlib import Path
import unicodedata
import re
from typing import Union, List
import numpy as np
from itertools import chain
import os

import torch
import torch.nn as nn
from torch.utils.data import BatchSampler

from fastNLP import Callback, logger, ReproducibleBatchSampler







def _init_weights(module):
    """Initialize the weights"""
    if isinstance(module, nn.Linear):
        # Slightly different from the TF version which uses truncated_normal for initialization
        # cf https://github.com/pytorch/pytorch/pull/5617
        module.weight.data.normal_(mean=0.0, std=0.02)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
        module.bias.data.zero_()
        module.weight.data.fill_(1.0)
    elif isinstance(module, nn.Conv1d):
        nn.init.kaiming_normal_(module.weight.data)
    if isinstance(module, (nn.Linear, nn.Conv1d)) and module.bias is not None:
        module.bias.data.zero_()


class LRSchedCallback(Callback):
    """
    根据 ``step_on`` 参数在合适的时机调用 scheduler 的 step 函数。

    :param scheduler: 实现了 :meth:`step` 函数的对象；
    :param step_on: 可选 ``['batch'， 'epoch']`` 表示在何时调用 scheduler 的 step 函数。如果为 ``batch`` 的话在每次更新参数
        之前调用；如果为 ``epoch`` 则是在一个 epoch 运行结束后调用；
    """
    def __init__(self, scheduler, begin_epochs: int = 0, step_on:str='batch', name: str = 'lr scheduler'):
        assert hasattr(scheduler, 'step') and callable(scheduler.step), "The scheduler object should have a " \
                                                                        "step function."
        self.scheduler = scheduler
        self.begin_epochs = begin_epochs
        self.step_on = 0 if step_on == 'batch' else 1

        self.name = name

    def on_after_optimizers_step(self, trainer, optimizers):
        if trainer.cur_epoch_idx >= self.begin_epochs:
            if self.step_on == 0:
                self.scheduler.step()

    def on_train_epoch_end(self, trainer):
        if trainer.cur_epoch_idx >= self.begin_epochs:
            if self.step_on == 1:
                self.scheduler.step()
                

    def callback_name(self):
        return self.name    
    
    
class ConstantTokenBatchSampler(ReproducibleBatchSampler):
    """
    实现类似于 fairseq 的按照 batch 内句子总长度来计算具体的 batch_size；
    
    """
    
    def __init__(self, seq_len, one_batch_total_tokens: int, need_be_multiple_of=1, num_buckets=-1, shuffle: bool = False,
                 seed: int = 0, dataset_name: str = 'tmp', **kwargs):
        
        logger.info(f"{dataset_name} 使用 ConstantTokenBatchSampler.")
        
        self.seq_len = seq_len

        self.one_batch_total_tokens = one_batch_total_tokens
        self.need_be_multiple_of = need_be_multiple_of
        self.num_buckets = num_buckets
        self.shuffle = shuffle
        self._seed = int(seed)
        self.dataset_name = dataset_name

        self.num_replicas = kwargs.get("num_replicas", 1)
        self.rank = kwargs.get("rank", 0)
        self.epoch = kwargs.get("epoch", -1)
        self.pad = kwargs.get("pad", False)  # 该参数在该 batch sampler 中不具备意义；

        self.during_iter = kwargs.get("during_iter", False)

        seq_len_indices = [(length, i) for i, length in enumerate(seq_len)]
        seq_len_indices.sort(key=lambda x: x[0])
        indices_in_buckets = []
        if num_buckets > 0:
            sample_per_bucket = len(seq_len_indices) // num_buckets
            
            for i in range(num_buckets):
                indices_in_buckets.append(seq_len_indices[i * sample_per_bucket: (i+1) * sample_per_bucket])
            indices_in_buckets[-1].extend(seq_len_indices[(i+1) * sample_per_bucket:])
        else:
            indices_in_buckets = [seq_len_indices]
        self.indices_in_buckets = indices_in_buckets

        self.batch_size = -1  # for fastNLP
        # self.batch_size = sum(map(len, self.batches[:_sample])) // _sample
        self.batches = self._batchify() 

    @property
    def seed(self):
        return abs(self._seed + self.epoch)

    @property
    def batches(self):
        return self._batches

    @batches.setter
    def batches(self, batches):
        self._batches = batches
    
    def _batchify(self):
        over_long_num = 0
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(self.indices_in_buckets)
            for bucket in self.indices_in_buckets:
                rng.shuffle(bucket)
        
        indices = list(chain(*self.indices_in_buckets))
        batches = []
        cur_max_len = 0
        batch = []
        for length, i in indices:
            max_len = max(length, cur_max_len)
            if max_len*(len(batch)+1) > self.one_batch_total_tokens:
                left_sample = len(batch) % self.need_be_multiple_of
                add_samples = batch.copy()
                cur_max_len = length
                if left_sample != 0:
                    add_samples = add_samples[:-left_sample]
                    batch = batch[-left_sample:]
                    cur_max_len = max(cur_max_len, max([indices[_i][0] for _i in batch]))
                else:
                    batch = []
                if len(add_samples)==0:
                    over_long_num += 1

                batches.append(add_samples)
            else:
                cur_max_len = max_len
            batch.append(i)
        if batch:
            left_sample = len(batch) % self.need_be_multiple_of
            add_samples = batch.copy()
            if left_sample != 0:
                add_samples = add_samples[:-left_sample].copy()
            if add_samples:
                batches.append(add_samples)
        if self.shuffle:
            rng = np.random.default_rng(self.seed)
            rng.shuffle(batches)

        # most token 放前面
        # 最长的放前面
        most_token_idx = np.argmax([sum([self.seq_len[b] for b in batch]) for batch in batches])
        most_length_idx = np.argmax(map(len, batches))
        if most_length_idx != most_token_idx:
            for idx in [most_token_idx, most_length_idx]:
                batch = batches.pop(idx)
                batches.insert(0, batch)
        else:
            batch = batches.pop(most_token_idx)
            batches.insert(0, batch)
        
        if over_long_num > 0:
            logger.warning(f"{self.dataset_name} 有 {over_long_num} 个数据超过最大长度的限制；")
        

        return batches        

    def __iter__(self):
        self.during_iter = True
        
        batches = self.batches[self.rank::self.num_replicas]
        for batch in batches:
            yield batch
        
        self.during_iter = False

        # 只拿到自己当前 rank 的 batches；
        if self.shuffle or (not self.shuffle and self.batches is None):
            self.batches = self._batchify()
            self.batches = self.batches[: len(self.batches) // self.num_replicas * self.num_replicas]
            _all_samples = sum(map(len, self.batches))
            logger.info(f'所有 rank 的样本数一共为 {_all_samples}， 每个 rank 上的 batch 数量为: {len(self.batches) // self.num_replicas}.')

        if self.epoch < 0:
            self.epoch -= 1

    def set_epoch(self, epoch):
        self.epoch = epoch

    def __len__(self):
        return len(self.batches) // self.num_replicas

    def set_distributed(self, num_replicas, rank, pad=True):
        """
        进行分布式的相关设置，应当在初始化该 BatchSampler 本身后立即被调用。

        :param num_replicas: 分布式训练中的进程总数
        :param rank: 当前进程的 ``global_rank``
        :param pad: 如果 sample 数量不整除 ``num_replicas`` 的时候，要不要 pad 一下，使得最终使得每个进程上
            的 sample 数量是完全一致的
        :return: 自身
        """
        assert self.during_iter is False, "Cannot set the sampler to be distributed when it is " \
                                          "during an unfinished iteration."
        assert num_replicas > 0 and isinstance(num_replicas, int)
        assert isinstance(rank, int) and 0 <= rank < num_replicas
        # 注意初始化该函数时，所有的状态都应当默认是一个 epoch 刚开始训练的状态；
        self.num_replicas = num_replicas
        self.rank = rank
        self.pad = pad

        self.batches = self.batches[: len(self.batches) // self.num_replicas * self.num_replicas]
        _all_samples = sum(map(len, self.batches))
        logger.info(f'所有 rank 的样本数一共为 {_all_samples}， 每个 rank 上的 batch 数量为: {len(self.batches) // self.num_replicas}.')    

        return self

    

        
class CudaStatCallback(Callback):
    
    
    def __init__(self) -> None:
        super().__init__()

        self._source_shape = None
        
    def on_after_trainer_initialized(self, trainer, driver):
        cur_device = torch.cuda.current_device()
        self.prop = torch.cuda.get_device_properties("cuda:{}".format(cur_device))
    
    def on_train_batch_begin(self, trainer, batch, indices):
        self._source_shape = batch['audio_input']['input_values'].shape

    def on_exception(self, trainer, exception):
        logger.error(f"最后一个 batch 的输入音频的shape是 {self._source_shape}.")
        self._print_gpu_used_info()        
        
    def _print_gpu_used_info(self):
        gb_used = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
        gb_free = self.prop.total_memory / 1024 / 1024 / 1024 - gb_used
        logger.info(f"gb used: {gb_used}, gb free: {gb_free}")
        
    

def _flatten_dir(path):
    # 遍历一个目录，返回最终所有的扁平目录；
    
    path = Path(path)
    return_list = []    
    if path.is_dir():
        not_sub_dir = True
        for _path in path.iterdir():
            if _path.is_dir():
                not_sub_dir = False
                sub_path_list = _flatten_dir(_path)
                return_list.extend(sub_path_list)
        if not_sub_dir:
            return_list.append(str(path))
    return return_list


def is_all_chinese(strs):
    for _char in strs:
        if not '\u4e00' <= _char <= '\u9fa5':
            return False
    return True


def is_all_english(strs):
    for _char in strs:
        if not (u'\u0041'<= _char <= u'\u005a') and not (u'\u0061'<= _char <= u'\u007a'):
            return False
    return True


def is_all_english_and_space(strs):
    for _char in strs:
        if not (u'\u0041'<= _char <= u'\u005a') and not (u'\u0061'<= _char <= u'\u007a') and _char != ' ':
            return False
    return True


def normalize_txt(txt: str):
    txt = unicodedata.normalize('NFKC', txt)
    patttern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]") # 匹配不是中文、大小写、数字的其他字符
    txt = patttern.sub('', txt) #将string1中匹配到的字符替换成空字符
    return txt.upper()


def process_txt_part(_txt_part):
    if is_all_chinese(_txt_part) or _txt_part.isdigit():
        res = list(_txt_part)
    else:
        _txt_part = normalize_txt(_txt_part)
        if is_all_chinese(_txt_part) or _txt_part.isdigit():
            res = list(_txt_part)
        elif is_all_english(_txt_part):
            res = [_txt_part]
        else:
            ii = 0
            jj = 0
            last_type = -1
            sub_used_txt = []
            while ii + jj < len(_txt_part):
                cur_char = _txt_part[ii + jj]
                if cur_char != ' ':
                    if is_all_chinese(cur_char):
                        cur_type = 0
                    elif is_all_english(cur_char):
                        cur_type = 1
                    else:
                        cur_type = 2
                else:
                    cur_type = last_type
                
                if last_type != -1 and last_type != cur_type:
                    cur_sub_txt_part = _txt_part[ii: ii + jj]
                    if last_type in {0, 2}:
                        cur_sub_txt_part = cur_sub_txt_part.replace(' ', '')
                        sub_used_txt.extend(list(cur_sub_txt_part))
                    else:
                        sub_used_txt.append(cur_sub_txt_part.strip())  # 去除头尾的空格；
                    ii = ii + jj
                    jj = 0
                last_type = cur_type
                jj += 1
            
            cur_sub_txt_part = _txt_part[ii: ii + jj]
            if last_type in {0, 2}:
                cur_sub_txt_part = cur_sub_txt_part.replace(' ', '')
                sub_used_txt.extend(list(cur_sub_txt_part))
            else:
                sub_used_txt.append(cur_sub_txt_part.strip())  # 去除头尾的空格；
            
            res = sub_used_txt
    return res


def transform_phones(phones: Union[int, List[int]], phones_dict):
    
    if isinstance(phones, (int, torch.Tensor)):
        return phones_dict[phones]
    else:
        res = []
        for idx in phones:
            res.append(phones_dict[idx])
        return res        
    
    
    




     