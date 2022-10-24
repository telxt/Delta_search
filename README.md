# 一些问题

- 数据集需要列上去吗
- 模型需要给吗
- 需不需要给出baseline的实现





# To do list

- 修改t5-train.py
- 最后删除result文件夹和__pycache__文件夹
- 修改一下.sh文件，排一下版





以下是正文（还要参考声鼎的完善一下）：



# Task-Level Transferability of Parameter-efficient Modules

## 环境配置

```
python==3.8.12
transformer==4.10.0
```

## 一些说明

### 测试下游任务效果的pipline

#### 示例命令

```bash
cd scripts
bash adapter.sh
```

####sh文件说明

scripts文件夹中提供了`adapter.sh`，`lora.sh`，`prefix.sh`；分别对应着使用不同的delta tuning方法。以`adapter.sh`为例，其中各项的具体意义如下：

TODO



###对source delta objects进行融合的pipline

#### 对所有任务平均/对人工划分的任务进行平均

##### 示例命令

```bash
cd unit_model
python unit_model_average.py
```

`unit_model_average.py`默认是对于adapter方法生成所有已有

