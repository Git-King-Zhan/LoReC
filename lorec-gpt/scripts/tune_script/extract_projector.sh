# to fill in the following path to extract projector for the second tuning stage!
src_model=./checkpoints/stage_1/checkpoint-67200
output_proj=./checkpoints/stage_1/graph_projector/checkpoint-67200.bin

python3.10 ./scripts/extract_graph_projector.py \
  --model_name_or_path ${src_model} \
  --output ${output_proj}
