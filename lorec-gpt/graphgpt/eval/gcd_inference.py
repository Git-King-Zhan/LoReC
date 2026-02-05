"""
Graph Contrastive Decoding (GCD) Inference Integration Module
This module provides utilities to integrate GCD into the inference pipeline.
It handles:
    1. Augmenting graph data
    2. Preparing augmented graph data for model input
    3. Managing GCD hyperparameters
    4. Applying GCD to the generation process
"""

import copy
from typing import Optional, List, Dict, Any, Union
import torch
from torch_geometric.data import Data

try:
    from gcd_augmentation import GraphAugmentor, create_augmented_graph_data, create_augmented_graph_data_list
except ImportError:
    # Fallback: define minimal versions
    class GraphAugmentor:
        def __init__(self, augmentation_type: str = 'degree', drop_edge_rate: float = 0.1):
            self.augmentation_type = augmentation_type
            self.drop_edge_rate = drop_edge_rate
        
        def augment(self, graph_data: Data) -> Data:
            # Simple uniform edge dropout as fallback
            edge_index = graph_data.edge_index
            num_edges = edge_index.shape[1]
            keep_mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge_rate
            augmented_edge_index = edge_index[:, keep_mask]
            return Data(x=graph_data.x, edge_index=augmented_edge_index)
    
    def create_augmented_graph_data(graph_data: Data, augmentation_type: str = 'degree', drop_edge_rate: float = 0.1) -> Data:
        augmentor = GraphAugmentor(augmentation_type, drop_edge_rate)
        return augmentor.augment(graph_data)
    
    def create_augmented_graph_data_list(graph_data_list: list, augmentation_type: str = 'degree', drop_edge_rate: float = 0.1) -> list:
        augmentor = GraphAugmentor(augmentation_type, drop_edge_rate)
        return [augmentor.augment(g) for g in graph_data_list]

try:
    from gcd_sample import evolve_gcd_sampling
except ImportError:
    # Fallback: no-op function
    def evolve_gcd_sampling():
        pass


class GCDConfig:
    """Configuration for Graph Contrastive Decoding."""
    
    def __init__(
        self,
        enable_gcd: bool = True,
        augmentation_type: str = 'degree',
        drop_edge_rate: float = 0.2,
        cd_alpha: float = 0.5,
        cg_beta: float = 1.0,
        cut_para: float = 1.0,
        use_text_only_contrast: bool = True,
        use_augmented_graph_contrast: bool = True,
        edge_threshold: int = 5,  # Add edge_threshold parameter

    ):
        """
        Initialize GCD configuration.
        
        Args:
            enable_gcd: Whether to enable GCD
            augmentation_type: Type of graph augmentation ('uniform', 'degree', 'pr', 'evc')
            drop_edge_rate: Probability of dropping each edge during augmentation
            cd_alpha: Hyperparameter for text-only contrast
            cg_beta: Hyperparameter for augmented graph contrast
            cut_para: Hyperparameter used ONLY for cutoff threshold. If None, runtime falls back to cg_beta
            use_text_only_contrast: Whether to use text-only contrast (requires graph_data_cd)
            use_augmented_graph_contrast: Whether to use augmented graph contrast (requires graph_data_cg)
        """
        self.enable_gcd = enable_gcd
        self.augmentation_type = augmentation_type
        self.drop_edge_rate = drop_edge_rate
        self.cd_alpha = cd_alpha
        self.cg_beta = cg_beta
        self.cut_para = cut_para
        self.use_text_only_contrast = use_text_only_contrast
        self.use_augmented_graph_contrast = use_augmented_graph_contrast
        self.edge_threshold = edge_threshold  


class GCDInference:
    """
    GCD Inference Manager.
    Handles augmentation and preparation of graph data for GCD-enhanced generation.
    """
    
    def __init__(self, config: Optional[GCDConfig] = None):
        """
        Initialize GCD inference manager.
        
        Args:
            config: GCDConfig instance. If None, uses default configuration.
        """
        self.config = config if config is not None else GCDConfig()
        self.augmentor = GraphAugmentor(
            augmentation_type=self.config.augmentation_type,
            drop_edge_rate=self.config.drop_edge_rate
        )
        
        # Enable GCD sampling
        if self.config.enable_gcd:
            evolve_gcd_sampling()
    
    def prepare_gcd_inputs(
        self,
        input_ids: torch.LongTensor,
        graph_data: Union[Data, List[Data]],
        model_kwargs: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Prepare inputs for GCD-enhanced generation.
        
        This function:
        1. Creates augmented graph data (graph_data_cg)
        2. Optionally creates text-only graph data (graph_data_cd)
        3. Adds GCD hyperparameters to model_kwargs
        
        Args:
            input_ids: Input token IDs
            graph_data: Original graph data (can be Data or list of Data)
            model_kwargs: Model keyword arguments
        
        Returns:
            Updated model_kwargs with GCD-specific inputs
        """
        if not self.config.enable_gcd:
            return model_kwargs
        
        # Make a copy to avoid modifying the original
        gcd_kwargs = copy.deepcopy(model_kwargs)
        
        # Store original graph data
        gcd_kwargs['graph_data'] = graph_data


        if not self.config.use_augmented_graph_contrast:
            gcd_kwargs['graph_data_cg'] = None
            gcd_kwargs['cg_beta'] = 0.0
        
        # Create augmented graph data for augmented graph contrast (robust + policy)
        if self.config.use_augmented_graph_contrast:
            def _n_nodes_edges(g):
                n_edges = g.edge_index.shape[1] if hasattr(g, 'edge_index') and g.edge_index is not None else 0
                if hasattr(g, 'x') and g.x is not None:
                    n_nodes = g.x.shape[0]
                elif hasattr(g, 'graph_node') and g.graph_node is not None:
                    n_nodes = g.graph_node.shape[0]
                else:
                    n_nodes = 0
                return n_nodes, n_edges

            EDGE_THRESHOLD = self.config.edge_threshold

            if isinstance(graph_data, list):
                disable_cg = False
                orig_e_list = []
                for g in graph_data:
                    n_nodes, n_edges = _n_nodes_edges(g)
                    orig_e_list.append(n_edges)
                    if n_edges == 0:
                        disable_cg = True
                        print(f"[GCD] Disable CG for batch: zero-edge graph detected (n_nodes={n_nodes}, n_edges={n_edges})")
                        break
                    if n_edges < EDGE_THRESHOLD:
                        disable_cg = True
                        print(f"[GCD] Disable CG for batch: edge_count<{EDGE_THRESHOLD} (n_edges={n_edges})")
                        break
                if not disable_cg:
                    try:
                        aug_list = [create_augmented_graph_data(
                            g,
                            augmentation_type=self.config.augmentation_type,
                            drop_edge_rate=self.config.drop_edge_rate
                        ) for g in graph_data]

                        for idx_g, ag in enumerate(aug_list):
                            _, e2 = _n_nodes_edges(ag)
                            e_orig = orig_e_list[idx_g]
                            if e2 == 0:
                                disable_cg = True
                                print("[GCD] Disable CG: augmented graph has zero edges")
                                break
                            if e2 == e_orig:
                                disable_cg = True
                                print(f"[GCD] Disable CG: augmented edge count unchanged (e={e2})")
                                break
                        if not disable_cg:
                            gcd_kwargs['graph_data_cg'] = aug_list
                        else:
                            gcd_kwargs['graph_data_cg'] = None
                            gcd_kwargs['cg_beta'] = 0.0
                    except Exception as e:
                        print(f"[GCD][Warn] Augmentation failed for batch: {e}. Disable CG branch.")
                        gcd_kwargs['graph_data_cg'] = None
                        gcd_kwargs['cg_beta'] = 0.0
                else:
                    gcd_kwargs['graph_data_cg'] = None
                    gcd_kwargs['cg_beta'] = 0.0
            else:
                n_nodes, n_edges = _n_nodes_edges(graph_data)
                if n_edges == 0:
                    print(f"[GCD] Disable CG: zero-edge graph (n_nodes={n_nodes}, n_edges={n_edges})")
                    gcd_kwargs['graph_data_cg'] = None
                    gcd_kwargs['cg_beta'] = 0.0
                elif n_edges < EDGE_THRESHOLD:
                    print(f"[GCD] Disable CG: edge_count<{EDGE_THRESHOLD} (n_edges={n_edges})")
                    gcd_kwargs['graph_data_cg'] = None
                    gcd_kwargs['cg_beta'] = 0.0
                else:
                    try:
                        ag = create_augmented_graph_data(
                            graph_data,
                            augmentation_type=self.config.augmentation_type,
                            drop_edge_rate=self.config.drop_edge_rate
                        )
                        _, e2 = _n_nodes_edges(ag)
                        if e2 == 0:
                            print("[GCD] Disable CG: augmented graph has zero edges")
                            gcd_kwargs['graph_data_cg'] = None
                            gcd_kwargs['cg_beta'] = 0.0
                        elif e2 == n_edges:
                            print(f"[GCD] Disable CG: augmented edge count unchanged (e={e2})")
                            gcd_kwargs['graph_data_cg'] = None
                            gcd_kwargs['cg_beta'] = 0.0
                        else:
                            gcd_kwargs['graph_data_cg'] = ag
                    except Exception as e:
                        print(f"[GCD][Warn] Augmentation failed: {e}. Disable CG branch.")
                        gcd_kwargs['graph_data_cg'] = None
                        gcd_kwargs['cg_beta'] = 0.0
        
        # For text-only contrast, we don't include graph data
        # This is handled by setting graph_data_cd to None in the model_kwargs
        # The model will skip graph processing when graph_data_cd is None
        if self.config.use_text_only_contrast:
            gcd_kwargs['graph_data_cd'] = None  # Signal to skip graph processing
        
        # Add GCD hyperparameters
        gcd_kwargs['cd_alpha'] = self.config.cd_alpha
        gcd_kwargs['cg_beta'] = self.config.cg_beta

        # NOTE: Do not pass extra keys not recognized by model.generate to avoid ValueError.
        # If you need any cutoff-like hyperparams, consume them inside GCDInference instead of passing through.
        
        return gcd_kwargs
    
    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        graph_data: Optional[Union[Data, List[Data]]] = None,
        **model_kwargs
    ) -> Dict[str, Any]:
        """
        Prepare inputs for generation with GCD support.
        
        This is a wrapper around prepare_gcd_inputs that handles the full preparation.
        
        Args:
            input_ids: Input token IDs
            graph_data: Original graph data
            **model_kwargs: Additional model arguments
        
        Returns:
            Prepared model_kwargs for generation
        """
        if graph_data is not None:
            model_kwargs = self.prepare_gcd_inputs(input_ids, graph_data, model_kwargs)
        
        return model_kwargs


def apply_gcd(
    model,
    input_ids: torch.LongTensor,
    graph_data: Optional[Union[Data, List[Data]]] = None,
    gcd_config: Optional[GCDConfig] = None,
    **generation_kwargs
) -> torch.LongTensor:
    """
    Convenience function to apply GCD to model generation.
    
    Args:
        model: The language model
        input_ids: Input token IDs
        graph_data: Graph data for the model
        gcd_config: GCD configuration
        **generation_kwargs: Additional generation arguments
    
    Returns:
        Generated token IDs
    """
    if gcd_config is None:
        gcd_config = GCDConfig()
    
    gcd_manager = GCDInference(gcd_config)
    
    # Prepare GCD inputs
    if graph_data is not None:
        generation_kwargs = gcd_manager.prepare_gcd_inputs(
            input_ids, graph_data, generation_kwargs
        )
    
    # Generate with GCD
    output_ids = model.generate(input_ids, **generation_kwargs)
    
    return output_ids


def create_gcd_model_kwargs(
    graph_data: Optional[Union[Data, List[Data]]] = None,
    gcd_config: Optional[GCDConfig] = None,
    **additional_kwargs
) -> Dict[str, Any]:
    """
    Create model_kwargs dictionary with GCD support.
    
    Args:
        graph_data: Graph data for the model
        gcd_config: GCD configuration
        **additional_kwargs: Additional arguments to include
    
    Returns:
        model_kwargs dictionary with GCD support
    """
    if gcd_config is None:
        gcd_config = GCDConfig()
    
    gcd_manager = GCDInference(gcd_config)
    
    model_kwargs = additional_kwargs.copy()
    
    if graph_data is not None:
        # Create dummy input_ids for prepare_gcd_inputs
        # This is just for initialization; actual input_ids will be used during generation
        dummy_input_ids = torch.zeros(1, 1, dtype=torch.long)
        model_kwargs = gcd_manager.prepare_gcd_inputs(dummy_input_ids, graph_data, model_kwargs)
    
    return model_kwargs

