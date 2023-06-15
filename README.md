# SGG_Attack
Adversarial Attacks on Scene Graph Generation

# Scene Graph Benchmark in PyTorch 1.7
- **This project is based on [maskrcnn-benchmark] and [scene graph benchmark]**
- **(Adv_BOX)[https://github.com/advboxes/AdvBox/blob/a4ecf3026aa3e125463c513f93c4e2abf92a5120/adversarialbox/attacks/gradient_method.py]**
- **Foll_Box = https://github.com/bethgelab/foolbox/blob/1c55ee4d6847247eb50f34dd361ed5cd5b5a10bb/foolbox/attacks/gradient_descent_base.py**

## Highlights
- **General SGG Attack Method**
- **Available for SGDet and SGCls**
- **Various Attack Methods**
- **Various SGG Models**





## Installation

Check [INSTALL.md](INSTALL.md) for installation instructions.


## Model Zoo and Baselines

Pre-trained models


## Perform training

For the following examples to work, you need to first install this repo.

You will also need to download the dataset. Datasets can be downloaded by [azcopy](https://docs.microsoft.com/en-us/azure/storage/common/storage-use-azcopy-v10) with following command:
```bash
path/to/azcopy copy 'https://penzhanwu2.blob.core.windows.net/sgg/sgg_benchmark/datasets/TASK_NAME' <target folder> --recursive
```
`TASK_NAME` could be `visualgenome`, `openimages_v5c`.

We recommend to symlink the path to the dataset to `datasets/` as follows

```bash
# symlink the dataset
cd ~/github/maskrcnn-benchmark
mkdir -p datasets/openimages_v5c/
ln -s /vrd datasets/openimages_v5c/vrd
```



### Single GPU training

```bash
python tools/train_sg_net.py --config-file "/path/to/config/file.yaml"
```
This should work out of the box and is very similar to what we should do for multi-GPU training.
But the drawback is that it will use much more GPU memory. The reason is that we set in the configuration files a global batch size that is divided over the number of GPUs. So if we only have a single GPU, this means that the batch size for that GPU will be 4x larger, which might lead to out-of-memory errors.


### Multi-GPU training
We use internally `torch.distributed.launch` in order to launch
multi-gpu training. This utility function from PyTorch spawns as many
Python processes as the number of GPUs we want to use, and each Python
process will only use a single GPU.

```bash
export NGPUS=4
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/train_sg_net.py --config-file "path/to/config/file.yaml" 
```


## Evaluation
You can test your model directly on single or multiple gpus. 
To evaluate relations, one needs to output "relation_scores_all" in the TSV_SAVE_SUBSET.
Here are a few example command line for evaluating on 4 GPUS:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH 

# vg IMP evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_imp.yaml

# vg MSDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_msdn.yaml

# vg neural motif evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_nm.yaml

# vg GRCNN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file sgg_configs/vg_vrd/rel_danfeiX_FPN50_grcnn.yaml

# vg RelDN evaluation
python -m torch.distributed.launch --nproc_per_node=$NGPUS toofls/test_sg_net.py --config-file sgg_conigs/vg_vrd/rel_danfeiX_FPN50_reldn.yaml

```

To evaluate in sgcls mode:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_BOX_HEAD.FORCE_BOXES True MODEL.ROI_RELATION_HEAD.MODE "sgcls"
```

To evaluate in predcls mode:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_RELATION_HEAD.MODE "predcls"
```

To evaluate with ground truth bbox and ground truth pairs:
```bash
export NGPUS=4

python -m torch.distributed.launch --nproc_per_node=$NGPUS tools/test_sg_net.py --config-file CONFIG_FILE_PATH MODEL.ROI_RELATION_HEAD.FORCE_RELATIONS True
```

## Perform attacking
To attack in sgdet/sgcls mode:
```bash
python od_adv_box/attack.py --config_file ~ --model_rel ~

python od_adv_box/attack_SGCls.py --config_file ~ --model_rel ~
```
Choose the available config_file from "./sgg_attack/sgg_configs" and set the model_rel to 'Motifs_Pred_Cls'/'Motifs_Pred_Cls'/'Reldn_Pred_Cls'/'Imp_Pred_Cls'/'Msdn_Pred_Cls'/'Grcnn_Pred_Cls'




## Troubleshooting


## Citations

Not available

  
## License

## Acknowledgement
