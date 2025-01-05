# prepare code
```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
```

# ENV
```bash
docker pull dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07
docker images
```
```
REPOSITORY                                                          TAG         IMAGE ID       CREATED        SIZE
nvcr.io/nvidia/pytorch                                              24.12-py3   eec0906cea58   4 weeks ago    21.7GB
dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch   24.07       98206e15e59a   3 months ago   21.2GB
```
```bash
docker run --gpus all --shm-size=8g -it -v /media/re/2384a6b4-4dae-400d-ad72-9b7044491b55/LLM_LR/Pai-Megatron-Patch:/workspace/shenxiao -p 3334:22 dsw-registry.cn-wulanchabu.cr.aliyuncs.com/pai/pai-megatron-patch:24.07
```

# Connect SSH
```bash
apt-get update
apt-get install openssh-server -y

vi /etc/ssh/sshd_config
PermitRootLogin yes  # 允许root登录
PasswordAuthentication yes  # 允许密码认证

passwd root

service ssh restart
service ssh status
```
```shell
Host Pai-Megatron-Patch-SSH
    HostName 127.0.0.1
    Port 3334
    User root
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
```


# Download Qwen2 Weights
```bash
mkdir Weights
cd Weights
mkdir qwen_ckpts
cd qwen_ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-ckpts/Qwen2-0.5B.tgz
tar -zxf Qwen2-0.5B.tgz
rm -r Qwen2-0.5B.tgz
```

# Download Qwen2 Dataset
```bash
mkdir Data
cd Data
mkdir wudao_qwenbpe_text_document
cd wudao_qwenbpe_text_document
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/wudao_qwenbpe_text_document.idx

```
```bash
mkdir alpaca_zh-qwen
cd alpaca_zh-qwen
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/qwen-datasets/alpaca_zh-qwen-valid.json
```

# Megatron-Core模型格式转换
```
hf2mcore_qwen2_convertor.sh
MODEL_SIZE=$1                  # 模型参数：0.5B/1.8B
SOURCE_CKPT_PATH=$2            # 源路径
TARGET_CKPT_PATH=$3            # 目标路径
TP=$4                          # 模型并行度
PP=$5                          # 流水并行度
EP=$6                          # 专家并行度
PR=$7                          # 转换精度
USE_TE=$8                      # 是否使用Transformer Engine建模
mg2hf=$9                       # 是否执行mcore2hf转换
HG_CKPT_PATH=${10}             # HF的CKPT的路径
```

将checkpoint转换为MCore-Dense格式

!!! hf2mcore_qwen2_convertor.sh 脚本修改
```
export CUDA_VISIBLE_DEVICES=0
```
```
!!! megatron_patch/model/qwen2/layer_specs.py 代码修改
```python
    from megatron.core.transformer.custom_layers.transformer_engine import (
        # TEColumnParallelGroupedLinear,
        TEDotProductAttention,
        TELayerNormColumnParallelLinear,
        TENorm,
        # TERowParallelGroupedLinear,
        TERowParallelLinear,
    )
```
```bash
cd /workspace/shenxiao/toolkits/model_checkpoints_convertor/qwen
```
```bash
bash hf2mcore_qwen2_convertor.sh \
0.5B \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
1  \
1  \
1 \
fp32 \
true \
false 
```

# Megatron-Core预训练及指令微调
run_mcore_qwen.sh
```bash
ENV=$1                          # 运行环境配置开关: dsw单机训练训练，dlc表示多机训练环境
MODEL_SIZE=$2                   # 模型结构参数量级: 7B, 72B, A14B
BATCH_SIZE=$3                   # 一次迭代一个数据并行内的样本数
GLOBAL_BATCH_SIZE=$4            # 一次迭代多个数据并行的总样本数
LR=$5                           # 学习率
MIN_LR=$6                       # 最小学习率
SEQ_LEN=$7                      # 序列长度
PAD_LEN=$8                      # Padding长度
PR=${9}                         # 训练精度: fp16, bf16, fp8
TP=${10}                        # 模型并行度
PP=${11}                        # 流水并行度
CP=${12}                        # 上下文并行度
EP=${13}                        # 专家并行度
SP=${14}                        # 是否使用序列并行: true, false
DO=${15}                        # 是否使用Megatron版Zero-1降显存优化器: true, false
FL=${16}                        # 是否优先使用Flash Attention: true, false
SFT=${17}                       # 是否执行微调训练: true, false
AC=${18}                        # 激活检查点模式: sel, full, offload, false
OPTIMIZER_OFFLOAD=${19}         # 是否启用Offload optimizer: false, static, auto
SAVE_INTERVAL=${20}             # 保存ckpt的间隔
DATASET_PATH=${21}              # 训练数据集路径
VALID_DATASET_PATH=${22}        # 验证数据集路径
PRETRAIN_CHECKPOINT_PATH=${23}  # 预训练模型路径
TRAIN_TOKENS_OR_ITERS=${24}     # 训练TOKEN或者Iter数
WARMUP_TOKENS_OR_ITERS=${25}    # 预热TOKEN或者Iter数        
OUTPUT_BASEPATH=${26}           # 训练输出日志文件路径
```
## 预训练
使用以下命令启动对qwen2的继续预训练。 备注：当AC=offload或full时，可设置MP_AC_LAYERS环境变量来控制Checkpointing或Offload的TransformerLayer层数（默认值：1）。
!!! examples/qwen2/run_mcore_qwen.sh 修改脚本
```shell
if [ $ENV = dsw ]; then
    # export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export CUDA_VISIBLE_DEVICES=0
    MASTER_ADDR=localhost
    MASTER_PORT=$(shuf -n 1 -i 10000-65535)
    NNODES=1
    NODE_RANK=0
    # GPUS_PER_NODE=8
    GPUS_PER_NODE=1
```
```bash
cd examples/qwen2
```
```bash
bash run_mcore_qwen.sh  \
dsw  \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
1 \
1 \
true \
true   \
true \
false \
false   \
false \
100000  \
/workspace/shenxiao/Data/wudao_qwenbpe_text_document/wudao_qwenbpe_text_document   \
/workspace/shenxiao/Data/wudao_qwenbpe_text_document/wudao_qwenbpe_text_document  \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
10000  \
100   \
/workspace/shenxiao/Results/qwen2_0.5b_pretrain_wudao_qwenbpe_text_document
```


## 指令微调
通过设置MP_DATASET_TYPE环境变量，本脚本还可使用json格式的数据集进行指令微调
```bash
export MP_DATASET_TYPE="raw"
```bash
bash run_mcore_qwen.sh  \
dsw  \
0.5B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
bf16  \
1   \
1  \
1 \
1 \
true \
true   \
true \
true \
false   \
false \
100000  \
/workspace/shenxiao/Data/alpaca_zh-qwen/alpaca_zh-qwen-train.json    \
/workspace/shenxiao/Data/alpaca_zh-qwen/alpaca_zh-qwen-valid.json  \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B-hf-to-mcore-te-tp1-pp1  \
10000  \
100   \
/workspace/shenxiao/Results/qwen2_0.5b_finetune_alpaca_zh-qwen
```


# Evaluate

## Evaluate Format Transform
将训练/微调后保存的Megatron-Core转换为HuggingFace格式来进行推理评估
```bash
cd /workspace/shenxiao/toolkits/model_checkpoints_convertor/qwen
```
```bash
bash hf2mcore_qwen2_convertor.sh \
0.5B \
/workspace/shenxiao/Results/qwen2_0.5b_finetune_alpaca_zh-qwen/checkpoint/finetune-mcore-qwen2-0.5B-lr-1e-5-minlr-1e-6-bs-1-gbs-8-seqlen-128-pr-bf16-tp-1-pp-1-cp-1-ac-false-do-true-sp-true-ti-1000-wi-100  \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B-mcore-te-to-hf_after_finetune_alpaca_zh-qwen    \
1  \
1  \
1 \
fp32 \
true \
true \
/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B
```

## Download Evaluate Dataset
```bash
cd /workspace

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/evaluate.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/cmmlu.tgz 
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/evaluation-datasets/ceval.tgz 


tar -xvzf cmmlu.tgz 
tar -xvzf ceval.tgz 
tar -xvzf evaluate.tgz
```

## Run Evaluate
```bash
cd /workspace/shenxiao/LM-Evaluation-Harness-240310
```
```bash
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/workspace/shenxiao/Weights/qwen_ckpts/Qwen2-0.5B-mcore-te-to-hf_after_finetune_alpaca_zh-qwen,trust_remote_code=True \
--tasks cmmlu,ceval-valid  \
--batch_size 16 \
--output_path /workspace/shenxiao/Results/qwen2_0.5b_finetune_alpaca_zh-qwen/evaluate_results
```

## Reference
https://github.com/alibaba/Pai-Megatron-Patch/tree/main/examples/qwen2