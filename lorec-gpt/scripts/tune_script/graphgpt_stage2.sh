model_path=./lorec-gpt/vicuna-7b-v1.5-16k
instruct_ds=./lorec-gpt/data/stage_2/arxiv_pub_node_st_cot_link_mix.json
graph_data_path=./lorec-gpt/graph_data/All_pyg_graph_data/all_graph_data.pt
pretra_gnn=clip_gt_arxiv
tuned_proj=./lorec-gpt/checkpoints/stage_1/arxivpub/graph_projector.bin
output_model=./lorec-gpt/checkpoints/stage_2/arxivpub_arxiv


wandb offline
python -m torch.distributed.run --nnodes=1 --nproc_per_node=4 --master_port=20002 \
    graphgpt/train/train_mem.py \
    --model_name_or_path ${model_path} \
    --version v1 \
    --data_path ${instruct_ds} \
    --graph_content ./arxiv_ti_ab.json \
    --graph_data_path ${graph_data_path} \
    --graph_tower ${pretra_gnn} \
    --pretrain_graph_mlp_adapter ${tuned_proj} \
    --tune_graph_mlp_adapter True \
    --graph_select_layer -2 \
    --use_graph_start_end True\
    --bf16 True \
    --output_dir ${output_model} \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 3000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --report_to wandb \
    --ddp_find_unused_parameters True
