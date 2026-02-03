# LoReC: Rethinking Large Language Models for Graph Data Analysis
![LoReC](assets/framework.png)
## :books: Overview
LoReC (Look, Remember, Contrast) is a a novel decoding method that comprehensively enhances LLMs’ understanding of graph data without extra fine-tuning. The contributions are as follows:
- LoReC significantly enhances GraphLLM models' perception and comprehension of graph data.
- LoReC is a plug-and-play solution without extra training or fine-tuning, enabling seamless integration with existing GraphLLM models.

## :briefcase: Datasets
```
# Get all the used datasets from the huggingface links given.

# All instruction dataset for evaluation.
[eval](https://huggingface.co/datasets/Jiabin99/GraphGPT-eval-instruction)

# All utilized graph data.
[All_pyg_graph_data](https://huggingface.co/datasets/Jiabin99/All_pyg_graph_data)

```
## :pushpin: Usage
### 1. Enviroment
You can install the required enviroment by running the following command:
```
conda create -n lorec-gpt python=3.10
conda activate lorec-gpt

# Install torch with cuda 11.8
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118

# Install pyg and pyg-relevant packages
pip install torch-geometric==2.6.1
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu118.html

# Install required libraries
pip install -r requirements.txt
```

### 2. Checkpoint
You can use the uploaded checkpoints or train from scratch.
```
# Train from scratch

# Stage-1
# Fill in the following paths in graphgpt_stage1.sh to conduct stage-1
model_path=./GraphGPT-main/vicuna-7b-v1.5-16k
instruct_ds=./GraphGPT-main/data/stage_1/graph_matching.json
graph_data_path=./GraphGPT-main/graph_data/All_pyg_graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
output_model=./GraphGPT-main/checkpoints/stage_1/arxivpub

# running stage-1
bash ./GraphGPT-main/scripts/tune_script/graphgpt_stage1.sh

# Stage-2
# Fill in the following paths in graphgpt_stage2.sh to conduct stage-2
model_path=./GraphGPT-main/vicuna-7b-v1.5-16k
instruct_ds=./GraphGPT-main/data/stage_2/arxiv_pub_node_st_cot_link_mix.json
graph_data_path=./GraphGPT-main/graph_data/All_pyg_graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
tuned_proj=./GraphGPT-main/checkpoints/stage_1/arxivpub/graph_projector.bin
output_model=./GraphGPT-main/checkpoints/stage_2/arxivpub_arxiv

# running stage-2
bash ./GraphGPT-main/scripts/tune_script/graphgpt_stage2.sh
```
### 3. Evaluation
```
# Fill in the following paths in graphgpt_eval.sh to conduct inference
output_model=path-to-arxivpub
datapath=path-to-arxiv_test_instruct_cot.json
graph_data_path=path-to-all_graph_data.pt
res_path=path-to-eval_output
start_id=0
end_id=20000
log_dir=path-to-logs

# An example is as follows:
output_model=/gemini/code/LoReC/GraphGPT-main/checkpoints/stage_2/arxivpub
datapath=/gemini/code/LoReC/GraphGPT-main/graph_data/eval/arxiv_test_instruct_cot.json
graph_data_path=/gemini/LoReC/code/GraphGPT-main/graph_data/All_pyg_graph_data/all_graph_data.pt
res_path=/gemini/code/LoReC/GraphGPT-main/eval_output/arxivpub_arxiv_test
start_id=0
end_id=20000
log_dir=/gemini/code/LoReC/GraphGPT-main/logs

# running eval
bash ./GraphGPT-main/scripts/eval_script/graphgpt_eval.sh
```
### 4. Calculate metrics
```
# Fill in the following paths in cal_metric_arxiv.py to calculate metrics
folder = 'path-to-evaloutput json files'
graph_data = th.load('path-to-all_graph_data.pt')['arxiv']
df = pd.read_csv('path-to-labelidx2arxivcategeory.csv')

# An example is as follows:
folder = '/gemini/code/LoReC/GraphGPT-main/eval_output/arxivpub_arxiv_test/arxivpub_arxiv_test_alpha0.5_beta1.0_drop0.2_edge10'
graph_data = th.load('/gemini/code/LoReC/GraphGPT-main/graph_data/All_pyg_graph_data/all_graph_data.pt')['arxiv']
df = pd.read_csv('/gemini/code/LoReC/GraphGPT-main/calculate_metric/labelidx2arxivcategeory.csv')

# running calculation
python path-to-cal_metric_arxiv.py
```









