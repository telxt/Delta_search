<div align="center">

<h1>Parameter-efficient Weight Ensembling Facilitates Task-level Knowledge Transfer</h1>

</div>

🎉  本项目为下面ACL2023论文的实现：[Parameter-efficient Weight Ensembling Facilitates Task-level Knowledge Transfer](https://aclanthology.org/2023.acl-short.24/)




## 环境配置

```
python==3.8.12
transformer==4.10.0
```



## 测试下游任务效果

### 示例命令

```bash
cd scripts
bash adapter.sh
```

### sh文件说明

scripts文件夹中提供了`adapter.sh`，`lora.sh`，`prefix.sh`；分别对应着使用不同的delta tuning方法。以`adapter.sh`为例，其中各项的具体意义如下：

- `do_train`：设置是否进行训练（train）
- `do_predict`：设置是否进行测试（test）
- `learning_rate_list`：设置学习率，若在此项中有多个值，则依次取各个值进行实验
- `bsz_list`：设置batch size，若在此项中有多个值，则依次取各个值进行实验
- `train_iters`：设置训练的最大步数
- `warmup_steps`：设置warm_up的步数
- `valid_interval`：设置在训练中进行valid的间隔步数
- `log_interval`：设置在训练中进行log记录的间隔步数
- `early_stop`：设置early_stop的最大步数，若在训练中valid结果连续该值的次数没有出现提升，则停止训练
- `predict_batch_size`：设置进行测试（test）时batch size的值
- `tune_method`：设置delta tune的方法
- `quiet`：设置是否将涉及数据处理的warnings打印
- `apply_adapter`：设置是否使用adapter
- `adapter_type`：设置adapter的类型
- `adapter_size`：设置adapter层的大小
- `model`：设置模型的路径，可以是本地路径或Huggingface transformer中的标准路径
- `tokenizer_path`：设置tokenizer的路径，可以是本地路径或Huggingface transformer中的标准路径
- `output_dir`：设置输出log和result的路径
- `task_dir`：设置任务数据的存储路径
- `train_checkpoint`：设置在训练时进行模型初始化的delta module的路径（在文件中未显示）
- `test_checkpoint`：设置在测试时进行模型初始化的delta module的路径（在文件中未显示）



## Investigated Methods

### Avg. of Checkpoints && Manual Division

#### 示例命令

```bash
cd unit_model
python unit_model_average.py
```

`unit_model_average.py`默认是对于adapter方法生成所有的已有模型的平均。代码第10至12行是参数设置，其中`delta`表示所选择的delta tuning方法的类型，在代码中提供`adapter`，`lora`，`prefix`这3种；`source_ckpt_path`待平均的模型的路径；`save_ckpt_path`是平均后模型的路径。

可以通过修改代码第6行的导入任务清单来实现对人工划分的任务对应的模型进行平均。例如`from task_list import machine_reading_comprehension as TASK_LIST`



### Parametric Efficient Weight Ensembling

以adapter及Loss方法为例，使用该方法**首先**需要计算任务之间的Loss相似度，具体的命令如下：

```bash
cd scripts/get_similarity
bash adapter_Loss.sh
```

运行完成后`adapter_Loss.sh`文件中给出的`output_dir`路径下会生成`result.tsv`文件，文件所存储的便是任务之间的Loss相似度。

**然后**需要根据相似度进行加权平均，具体的命令如下：

```bash
cd ../../unit_model
python unit_model_others.py
```

其中`unit_model_others.py`文件中第12至15行是待设置的参数：`tem`设置进行softmax时的温度大小；`source_ckpt_path`待平均的模型的路径；`save_ckpt_path`是平均后模型的路径，经过这一步骤便生成了更优的用于初始化的模型，以及显示各`source_ckpt_path`参与平均的权重的热力图。

**最后**利用得到的模型进行初始化，在下游任务上进行测试，具体命令如下：

```bash
cd ../scripts
bash adapter.sh
```

`KL-divergence`、`EL2N`、`Cosine of Logits and Labels`、`GraNd`方法的pipline与`Loss`相同，只需①将计算相似度的命令中的`.sh`文件修改为该方法对应的文件；②在根据相似度进行加权平均时修改`unit_model_others.py`文件的待设置的参数(代码11至14行)。

> 对于`KL-divergence`方法需要提前使用少量数据（实验中设置为64_tune）在随机初始化的条件下进行tune，将得到的各任务tune后的checkpoint放在`adapter_KL.sh`random_tuned_ckpt_path对应的路径下。



### Approaches Extracting Information from the Weights

#### Cosine

使用该方法需要额外用到：使用目标任务数据（默认少量）进行tune后得到的checkpoint，该步骤可以通过**测试下游任务效果的pipline**中的示例命令实现（在测试的同时会默认保存训练得到的checkpoint）。

##### 示例命令

```bash
cd unit_model
python unit_model_cos.py
```

`unit_model_cos.py`默认是对于adapter方法生成所有的已有模型的平均。代码第11至14行是参数设置，其中`temperature`设置进行softmax时的温度大小；`tuned_64shot_ckpt_path`是使用目标任务数据（默认少量）进行tune后得到的checkpoint的路径；`source_ckpt_path`待平均的模型的路径；`save_ckpt_path`是平均后模型的路径。

#### Euclidean

以adapter为例，使用该方法**首先**需要对source delta objects进行若干步数的tune，具体命令如下：

```bash
cd scripts
bash adapter_tuneall.sh
```

**然后**需要计算source delta object在tuned前后的欧氏距离，并据此对模型进行加权平均，具体的命令如下：

```bash
cd ../unit_model
python unit_model_Euclidean.py
```

**最后**利用得到的模型进行初始化，在下游任务上进行测试，具体命令如下：

```bash
cd ../scripts
bash adapter.sh
```

#### Performance

以adapter为例，使用该方法**首先**需要对source delta objects进行若干步数的tune，并且测试source delta objects在dev数据集上进行zero_shot的结果，具体命令如下：

```bash
cd scripts
bash adapter_tuneall.sh
bash adapter_devall.sh
```

**然后**需要计算source delta object在tuned前后的dev_performance变化，并据此对模型进行加权平均，具体的命令如下：

```bash
cd ../unit_model
python unit_model_Performance.py
```

**最后**利用得到的模型进行初始化，在下游任务上进行测试，具体命令如下：

```bash
cd ../scripts
bash adapter.sh
```



## Analysis of Module Importance

###  Modified GraNd Approach

以“模型粒度”是block的实验为例，**首先**需要依据24个block计算任务之间的GraNd相似度，具体的命令如下：

```bash
# 在运行命令前，需要将get_similarity/get_GraNd.py文件281行的mode修改为'block'
cd scripts/get_similarity
bash adapter_GraNd.sh
```

运行完成后`adapter_GraNd.sh`文件中给出的`output_dir`路径下会生成24个`result.tsv`文件，文件所存储的分别是依据24个block计算出的任务之间的GraNd相似度。

**然后**需要根据这24个相似度进行24次加权平均，具体的命令如下：

```bash
# 在运行命令前，应修改unit_model_others.py文件中第13行，使origin_data_path分别是此前得到的24个.tsv文件
cd ../../unit_model
python unit_model_others.py
```

**最后**利用得到的模型进行初始化，在下游任务上进行测试，具体命令如下：

```bash
cd ../scripts
bash adapter.sh
```

> 若要进行“模型粒度”是layer的实验，则需要将`get_similarity/get_GraNd.py`文件281行的`mode`修改为'layer'。



### Modified Cosine of Logits and Labels Approach

**首先**需要依据24个block计算任务之间的Cosine of Logits and Labels相似度，具体的命令如下：

```bash
cd scripts/get_similarity
bash adapter_block.sh
```

运行完成后`adapter_block.sh`文件中给出的`output_dir`路径下会生成24个`result.tsv`文件，文件所存储的分别是依据24个block计算出的任务之间的Cosine of Logits and Labels相似度。

**然后**需要根据这24个相似度进行24次加权平均，具体的命令如下：

```bash
# 在运行命令前，应修改unit_model_others.py文件中第13行，使origin_data_path分别是此前得到的24个.tsv文件
cd ../../unit_model
python unit_model_others.py
```

**最后**利用得到的模型进行初始化，在下游任务上进行测试，具体命令如下：

```bash
cd ../scripts
bash adapter.sh
```

## 说明

仓库中的部分代码（例如：tune_hps_singletask.py）参考了[Crossfit](https://github.com/INK-USC/CrossFit).

## 反馈问题或疑问？

如果您对该代码或论文有任何疑问，请联系 Xingtai Lv (lvxt20@mails.tsinghua.edu.cn) 或者开一个 Github issue。

## 引用

如果你觉得我们的工作有用，请参考以下引用：

```bibtex
@inproceedings{lv2023parameter,
  title={Parameter-efficient Weight Ensembling Facilitates Task-level Knowledge Transfer},
  author={Lv, Xingtai and Ding, Ning and Qin, Yujia and Liu, Zhiyuan and Sun, Maosong},
  booktitle={Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 2: Short Papers)},
  pages={270--282},
  year={2023}
}
```
