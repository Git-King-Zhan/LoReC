import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import os
from graphgpt.conversation import conv_templates, SeparatorStyle
from graphgpt.utils import disable_torch_init
from transformers import CLIPVisionModel, CLIPImageProcessor, StoppingCriteria
from graphgpt.model import *
from graphgpt.model.utils import KeywordsStoppingCriteria
from torch_geometric.data import Data
import json
import copy

import os
import requests
from PIL import Image
from io import BytesIO

from tqdm import tqdm
import json
import os.path as osp
import math

import ray
import time

from graphgpt.model.GraphLlama import GraphLlamaConfig, GraphLlamaForCausalLM
from graphgpt.eval.gcd_inference import GCDConfig, GCDInference

# os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'

# graph_reinjecting
from graph_reinjecting import apply_graphreinject
# amplifying_attention
from graph_amplifying_attention import apply_graph_attention_boost

DEFAULT_GRAPH_TOKEN = "<graph>"
DEFAULT_GRAPH_PATCH_TOKEN = "<g_patch>"
DEFAULT_G_START_TOKEN = "<g_start>"
DEFAULT_G_END_TOKEN = "<g_end>"


def load_graph(instruct_item, graph_data_path): 
    graph_data_all = torch.load(graph_data_path)
    graph_dict = instruct_item['graph']
    graph_edge_index = torch.Tensor(copy.deepcopy(graph_dict['edge_index'])).long()
    graph_node_list = copy.deepcopy(graph_dict['node_list'])
    target_node = copy.deepcopy(graph_dict['node_idx'])
    graph_type = copy.deepcopy(instruct_item['id']).split('_')[0]
    graph_node_rep = graph_data_all[graph_type].x[graph_node_list] # Retrieve the representation of the current graph node from all graph data.
    
    cur_token_len = len(graph_node_rep)   # FIXME: 14 is hardcoded patch size

    graph_ret = Data(graph_node = graph_node_rep, edge_index=graph_edge_index, target_node = torch.tensor([target_node]))
    # Ensure PyG compatibility: Mirror to x field
    graph_ret.x = graph_ret.graph_node

    return {
        'graph_data': graph_ret, 
        'graph_token_len': cur_token_len
    }


def load_prompting_file(file_path): 
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data


def run_eval(args, num_gpus):
    # The samples are evenly divided according to num_gpus (ceil, to ensure that the last slice gets the remainder).
    prompt_file = load_prompting_file(args.prompting_file)
    prompt_file = prompt_file[args.start_id:args.end_id]
    total = len(prompt_file)
    if total == 0:
        print("[RunEval] No samples to run. Exit.")
        return

    if num_gpus <= 0:
        num_gpus = 1

    chunk_size = math.ceil(total / num_gpus)
    if chunk_size <= 0:
        chunk_size = 1

    if osp.exists(args.output_res_path) is False:
        os.makedirs(args.output_res_path, exist_ok=True)

    print(f"[Task Distribution] Total samples: {total}, num_gpus: {num_gpus}, chunk_size: {chunk_size}")

    ans_handles = []
    created_tasks = 0
    for task_id in range(num_gpus):
        start_off = task_id * chunk_size
        end_off = min((task_id + 1) * chunk_size, total)
        if start_off >= end_off:
            print(f"[Task {task_id}] skipped (empty slice)")
            continue

        start_split = args.start_id + start_off
        end_split = args.start_id + end_off

        print(f"[Task {task_id}] Assign samples [{start_split}, {end_split}) count={end_off - start_off}")
        handle = eval_model.remote(
            args,
            prompt_file[start_off:end_off],
            start_split,
            end_split,
        )
        ans_handles.append(handle)
        created_tasks += 1

    print(f"[Task Distribution] Created {created_tasks} tasks")

    ans_jsons = []
    for i, ans_handle in enumerate(ans_handles):
        print(f"[Waiting] Task {i} to complete...")
        res = ray.get(ans_handle)
        print(f"[Completed] Task {i}, got {len(res)} results")
        ans_jsons.extend(res)

    # with open(args.output_res_path, "w") as ans_file:
    #     for line in ans_jsons:
    #         ans_file.write(json.dumps(line) + "\n")


@ray.remote(num_gpus=1)
@torch.inference_mode()
def eval_model(args, prompt_file, start_idx, end_idx):
    try:
        if torch.cuda.is_available():
            torch.cuda.set_device(0)
            cur = torch.cuda.current_device()
            print(f"[Worker] CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES','')} | current_device={cur} | name={torch.cuda.get_device_name(cur)} | handling [{start_idx}, {end_idx})")
    except Exception as e:
        print(f"[Worker][Warn] GPU setup failed: {e}")

    # load prompting file
    # prompt_file = load_prompting_file(args.prompting_file)

    # Model
    disable_torch_init()
    # model_name = os.path.expanduser(args.model_name)
    print('start loading')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    print('finish loading')
    print('start loading')
    config = GraphLlamaConfig.from_pretrained(args.model_name)
    model = GraphLlamaForCausalLM.from_pretrained(args.model_name, config=config, torch_dtype=torch.float16, use_cache=True, low_cpu_mem_usage=True).cuda()
    print('finish loading')

    if args.enable_graphreinject:
        print("Enabling GraphReinject")
        apply_graphreinject(
            model,
            starting_layer=args.graph_vr_start_layer,
            ending_layer=args.graph_vr_end_layer,
            entropy_threshold=args.graph_vr_entropy_thresh,
            retracing_ratio=args.graph_vr_retracing_ratio
        )

    use_graph_start_end = getattr(model.config, "use_graph_start_end", False)
    tokenizer.add_tokens([DEFAULT_GRAPH_PATCH_TOKEN], special_tokens=True)
    if use_graph_start_end:
        tokenizer.add_tokens([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN], special_tokens=True)
    # The vocabulary embedding matrix of the synchronous expansion model is used to avoid index out-of-bounds errors.
    model.resize_token_embeddings(len(tokenizer))
    # Thesaurus/Embedding Consistency Self-Check）
    try:
        emb_rows = model.get_input_embeddings().weight.shape[0]
        print(f"[Sanity] len(tokenizer)={len(tokenizer)}, emb_rows={emb_rows}")
        assert emb_rows == len(tokenizer), "The number of lines in the embedding does not match the word list length."
        g_patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
        assert g_patch_id != tokenizer.unk_token_id, "<g_patch> was not registered correctly, resulting in an unk id."
        if use_graph_start_end:
            g_start_id, g_end_id = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
            assert g_start_id != tokenizer.unk_token_id and g_end_id != tokenizer.unk_token_id, "<g_start>/<g_end> were not registered correctly, resulting in an unk id."
    except Exception as e:
        print(f"[Sanity][Warn] Vocabulary/embedding check failed: {e}")

    graph_tower = model.get_model().graph_tower
    
    # TODO: add graph tower
    # if graph_tower.device.type == 'meta':
    #     print('meta')
    clip_graph, args_graph= load_model_pretrained(CLIP, './clip_gt_arxiv')
    graph_tower = graph_transformer(args_graph)
    graph_tower = transfer_param_tograph(clip_graph, graph_tower) # Transfer the pre-trained parameters of the CLIP model (clip_graph) to the graph encoder (graph_tower).
    model.get_model().graph_tower = graph_tower.cuda()

    # transform token to id
    graph_tower.to(device='cuda', dtype=torch.float16)
    graph_config = graph_tower.config
    graph_config.graph_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_GRAPH_PATCH_TOKEN])[0]
    graph_config.use_graph_start_end = use_graph_start_end
    if use_graph_start_end:
        graph_config.graph_start_token, graph_config.graph_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
    # TODO: add graph token len

    # define graph_patch_id and graph_start_end_id(create)
    graph_patch_id = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
    graph_start_id = None
    graph_end_id = None
    if use_graph_start_end:
        graph_start_id, graph_end_id = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])

    # "Look"
    if args.enable_graph_attention_boost:
        print("Enabling Graph Attention Boost")
        apply_graph_attention_boost(
            model,
            starting_layer=args.graph_attn_start_layer,
            ending_layer=args.graph_attn_end_layer,
            entropy_threshold=args.graph_attn_entropy_thresh,
            alpha=args.graph_attn_alpha,
            graph_patch_id=graph_patch_id,
            graph_start_id=graph_start_id,
            graph_end_id=graph_end_id,
        )


    # ============ GCD (Graph Contrastive Decoding) ============
    if args.enable_gcd:
        gcd_config = GCDConfig(
            enable_gcd=True,
            augmentation_type='degree',
            drop_edge_rate=args.gcd_drop_edge_rate,
            cd_alpha=args.gcd_cd_alpha,
            cg_beta=args.gcd_cg_beta,
            cut_para=args.gcd_cut_para,
            use_text_only_contrast=True,
            use_augmented_graph_contrast=True,
            edge_threshold=args.gcd_edge_threshold,
        )
        gcd_manager = GCDInference(gcd_config)
        print(f"[GCD] initialized，config: augmentation_type={gcd_config.augmentation_type}, drop_edge_rate={gcd_config.drop_edge_rate}, cd_alpha={gcd_config.cd_alpha}, cg_beta={gcd_config.cg_beta}, edge_threshold={gcd_config.edge_threshold}")
    else:
        gcd_manager = None

    res_data = []
    print(f'total: {len(prompt_file)}')
    # Traverse each node
    for idx, instruct_item in tqdm(enumerate(prompt_file)):
        graph_dict = load_graph(instruct_item, args.graph_data_path) # load graph data
        graph_token_len = graph_dict['graph_token_len']
        graph_data = graph_dict['graph_data']
        qs = instruct_item["conversations"][0]["value"] # original text
        if use_graph_start_end:
            replace_token = DEFAULT_G_START_TOKEN + (DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len) + DEFAULT_G_END_TOKEN
        else:
            replace_token = DEFAULT_GRAPH_PATCH_TOKEN * graph_token_len
        qs = qs.replace(DEFAULT_GRAPH_TOKEN, replace_token)  # Replace placeholders with a sequence of graph tokens.

        conv_mode = "graphchat_v1"

        if args.conv_mode is not None and conv_mode != args.conv_mode:
            print('[WARNING] the auto inferred conversation mode is {}, while `--conv-mode` is {}, using {}'.format(conv_mode, args.conv_mode, args.conv_mode))
        else:
            args.conv_mode = conv_mode

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], qs)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        inputs = tokenizer([prompt])
        input_ids = torch.as_tensor(inputs.input_ids).cuda()

        # Ensure that start/end are perfectly aligned with <g_patch>*graph_token_len.
        try:
            ids = input_ids[0]
            pid = tokenizer.convert_tokens_to_ids(DEFAULT_GRAPH_PATCH_TOKEN)
            if use_graph_start_end:
                sid, eid = tokenizer.convert_tokens_to_ids([DEFAULT_G_START_TOKEN, DEFAULT_G_END_TOKEN])
                starts = (ids == sid).nonzero(as_tuple=False).squeeze(-1)
                ends = (ids == eid).nonzero(as_tuple=False).squeeze(-1)
                masked = (ids == pid).nonzero(as_tuple=False).squeeze(-1)
                assert starts.numel() == ends.numel() and starts.numel() >= 1, "<g_start>/<g_end> not match"
                s = int(starts[0].item()); e = int(ends[0].item())
                assert e > s + 1, "graph length invalid"
                assert (e - s - 1) == graph_token_len, f"graph length not equal to graph_token_len: {(e - s - 1)} vs {graph_token_len}"
                expected = torch.arange(s + 1, e, device=ids.device)
                actual = masked[(masked >= (s + 1)) & (masked < e)]
                assert actual.numel() == graph_token_len
                assert torch.equal(actual, expected)
            else:
                masked = (ids == pid).nonzero(as_tuple=False).squeeze(-1)
                assert masked.numel() == graph_token_len
                if masked.numel() > 0:
                    expected = torch.arange(int(masked[0].item()), int(masked[0].item()) + graph_token_len, device=ids.device)
                    assert torch.equal(masked, expected)
        except Exception as e:
            print(f"[Sanity][Seq] Sequence consistency check failed: {e}")
            raise

        stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)
        graph_data.graph_node = graph_data.graph_node.to(torch.float16)


        # ============ GCD input ============
        model_kwargs = {'graph_data': graph_data.cuda()}
        if gcd_manager is not None:
            model_kwargs = gcd_manager.prepare_gcd_inputs(
                input_ids=input_ids,
                graph_data=graph_data,
                model_kwargs=model_kwargs
            )

        with torch.inference_mode():
            generate_start = time.time()
            output_ids = model.generate(
                input_ids,
                **model_kwargs, # graphreinject_attn_gcd
                # graph_data=graph_data.cuda(), # base
                do_sample=True,
                temperature=0.2,
                max_new_tokens=1024,
                stopping_criteria=[stopping_criteria],
            )
            generate_time = time.time() - generate_start

        input_token_len = input_ids.shape[1]
        # Calculate the number and rate of newly generated tokens in this iteration.
        n_new_tokens = int(output_ids.shape[1] - input_token_len)
        per_token_time_ms = (generate_time * 1000.0) / max(n_new_tokens, 1)
        tokens_per_sec = (n_new_tokens / generate_time) if generate_time > 0 else float('inf')

        n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
        if n_diff_input_output > 0:
            print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')

        gen_ids = output_ids[0, input_token_len:]
        gen_ids_1024 = gen_ids[:min(gen_ids.shape[0], 1024)]
        gen_text_full_raw = tokenizer.decode(gen_ids, skip_special_tokens=False)
        gen_text_full_clean = tokenizer.decode(gen_ids, skip_special_tokens=True)
        gen_text_1024_raw = tokenizer.decode(gen_ids_1024, skip_special_tokens=False)
        gen_text_1024_clean = tokenizer.decode(gen_ids_1024, skip_special_tokens=True)

        outputs = tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]
        outputs = outputs.strip()
        if outputs.endswith(stop_str):
            outputs = outputs[:-len(stop_str)]
        outputs = outputs.strip()

        res_data.append({
            "id": instruct_item["id"],
            "node_idx": instruct_item["graph"]["node_idx"],
            "res": outputs,
            "n_new_tokens": n_new_tokens,
            "tokens_per_sec": tokens_per_sec,
            "per_token_time_ms": per_token_time_ms,
            "gen_text_full_raw": gen_text_full_raw,
            "gen_text_full_clean": gen_text_full_clean,
            "gen_text_1024_raw": gen_text_1024_raw,
            "gen_text_1024_clean": gen_text_1024_clean,
        }.copy())

        
        # folder_name / file_name can be configured via CLI
        # - output_folder_name: subfolder under output_res_path
        # - output_file_prefix: prefix for json filename
        folder_name = args.output_folder_name or f"default_eval"
        target_folder = osp.join(args.output_res_path, folder_name)
        os.makedirs(target_folder, exist_ok=True)
        print('successfully make dir:', target_folder)

        file_prefix = args.output_file_prefix or folder_name
        file_name = f"{file_prefix}_{start_idx}_{end_idx}.json"
        file_path = osp.join(target_folder, file_name)

        with open(file_path, "w") as fout:
            json.dump(res_data, fout, indent=4)

        print(f"\n--- Iteration {idx} Timing Analysis ---")
        print(f"  Model Generation:  {generate_time:.4f} s")
        print(f"  Generated tokens: {n_new_tokens}  |  {tokens_per_sec:.2f} tok/s  |  {per_token_time_ms:.2f} ms/tok")
        print(f"------------------------------------\n")    

    return res_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--prompting_file", type=str, default=None)
    parser.add_argument("--conv-mode", type=str, default=None)
    parser.add_argument("--graph_data_path", type=str, default=None)
    parser.add_argument("--cut-para", type=float, default=0.1, help="GCD cutoff coefficient")

    # Output naming (subfolder + filename prefix)
    parser.add_argument("--output-folder-name", type=str, default=None, help="Subfolder name under output_res_path")
    parser.add_argument("--output-file-prefix", type=str, default=None, help="Filename prefix (without suffix)")

    parser.add_argument("--output_res_path", type=str, default=None)
    parser.add_argument("--num_gpus", type=int, default=4)

    parser.add_argument("--start_id", type=int, default=0)
    parser.add_argument("--end_id", type=int, default=20000)

    # GCD parameters
    parser.add_argument("--enable-gcd", action='store_true', help="Enable Graph Contrastive Decoding")
    parser.add_argument("--gcd-drop-edge-rate", type=float, default=0.2, help="GCD drop edge rate")
    parser.add_argument("--gcd-cd-alpha", type=float, default=0.5, help="GCD text contrast weight")
    parser.add_argument("--gcd-cg-beta", type=float, default=1.0, help="GCD graph contrast weight")
    parser.add_argument("--gcd-cut-para", type=float, default=1.0, help="GCD cutoff coefficient")
    parser.add_argument("--gcd-edge-threshold", type=int, default=10, help="GCD edge threshold")

    # Graph Attention parameters
    parser.add_argument("--enable-graph-attention-boost", action='store_true', help="Enable Graph Attention Boost")
    parser.add_argument("--graph-attn-start-layer", type=int, default=15, help="Graph Attention starting layer")
    parser.add_argument("--graph-attn-end-layer", type=int, default=22, help="Graph Attention ending layer")
    parser.add_argument("--graph-attn-entropy-thresh", type=float, default=0.75, help="Graph Attention entropy threshold")
    parser.add_argument("--graph-attn-alpha", type=float, default=0.2, help="Graph Attention alpha value")

    # GraphReinject parameters
    parser.add_argument("--enable-graphreinject", action='store_true', help="Enable GraphReinject")
    parser.add_argument("--graph-reinject-start-layer", type=int, default=8, help="GraphReinject starting layer")
    parser.add_argument("--graph-reinject-end-layer", type=int, default=16, help="GraphReinject ending layer")
    parser.add_argument("--graph-reinject-entropy-thresh", type=float, default=0.75, help="GraphReinject entropy threshold")
    parser.add_argument("--graph-reinject-retracing-ratio", type=float, default=0.25, help="GraphReinject retracing ratio (alpha)")

    args = parser.parse_args()

    # eval_model(args)
    
    # ============ Ensure Ray is initialized correctly. ============
    
    if ray.is_initialized():
        print("[Main] Ray already initialized, shutting down...")
        ray.shutdown()
    torch.cuda.empty_cache()
    import time
    time.sleep(2)
    ray.init(num_gpus=args.num_gpus, ignore_reinit_error=True)
    resources = ray.available_resources()
    print(f"[Main] Ray resources: {resources}")
    
    if 'GPU' in resources:
        actual_gpus = int(resources['GPU'])
        print(f"[Main] Ray successfully allocated {actual_gpus} GPUs (requested: {args.num_gpus})")
        if actual_gpus < args.num_gpus:
            print(f"[WARNING] Only {actual_gpus} GPUs allocated out of {args.num_gpus} requested!")
            print("[WARNING] This might be due to:")
            print("  1. GPU memory insufficient")
            print("  2. Other processes using GPU")
            print("  3. GPU driver issues")
    else:
        print("[ERROR] Ray did not allocate any GPUs!")
    
    print("[Main] Starting evaluation...")
    run_eval(args, args.num_gpus)

