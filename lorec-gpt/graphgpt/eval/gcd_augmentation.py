"""
Graph Contrastive Decoding (GCD) Augmentation Module

This module implements graph augmentation strategies for GCD.
It provides three edge augmentation methods:
    1. Uniform edge dropout: randomly drop edges with uniform probability
    2. Degree-based edge dropout: drop edges based on node degree
    3. PageRank-based edge dropout: drop edges based on PageRank centrality
"""

import torch
from torch_geometric.utils import degree, to_undirected
from torch_geometric.data import Data
from typing import Optional, Tuple, Union

try:
    from gca_functional import (
        drop_edge_weighted,
        degree_drop_weights,
        pr_drop_weights,
        evc_drop_weights,
    )
except ImportError:
    # Fallback: define basic functions if gca_functional is not available
    from torch_geometric.utils import dropout_adj
    
    def drop_edge_weighted(edge_index, edge_weights, p: float, threshold: float = 1.):
        """Drop edges based on weights."""
        # Handle empty edges or weights safely
        if edge_index is None or edge_index.numel() == 0:
            return edge_index
        if edge_weights is None or edge_weights.numel() == 0:
            return edge_index
        # Normalize and clip
        mean_w = edge_weights.mean()
        if not torch.isfinite(mean_w) or mean_w <= 0:
            # Fallback to uniform keep
            return edge_index
        edge_weights = edge_weights / mean_w * p
        edge_weights = torch.where(edge_weights < threshold, edge_weights, torch.ones_like(edge_weights) * threshold)
        keep_prob = 1.0 - edge_weights
        # keep_prob = edge_weights
        keep_prob = torch.clamp(keep_prob, 0.0, 1.0)
        sel_mask = torch.bernoulli(keep_prob).to(torch.bool)
        if sel_mask.numel() == 0:
            return edge_index
        return edge_index[:, sel_mask]
    
    def degree_drop_weights(edge_index):
        """Compute degree-based drop weights (robust to empty/degenerate graphs)."""
        if edge_index is None or edge_index.numel() == 0:
            # No edges
            device = edge_index.device if edge_index is not None else 'cpu'
            return torch.ones(0, device=device)
        # Ensure proper device
        device = edge_index.device
        # Make undirected for degree calc
        edge_index_ = to_undirected(edge_index)
        if edge_index_.numel() == 0:
            return torch.ones(0, device=device)
        # Degree of target nodes
        deg = degree(edge_index_[1])
        if edge_index.size(1) == 0:
            return torch.ones(0, device=device)
        # Gather degrees of column nodes
        deg_col = deg[edge_index[1]].to(torch.float32)
        if deg_col.numel() == 0:
            return torch.ones(0, device=device)
        # Avoid log(0)
        s_col = torch.log(torch.clamp_min(deg_col, 1.0))
        # Compute normalized inverted weights; if denom ~ 0, fallback to ones
        s_max = torch.max(s_col) if s_col.numel() > 0 else torch.tensor(0.0, device=device)
        s_mean = torch.mean(s_col) if s_col.numel() > 0 else torch.tensor(0.0, device=device)
        denom = s_max - s_mean
        if not torch.isfinite(denom) or abs(float(denom)) < 1e-6:
            return torch.ones_like(s_col)
        weights = (s_max - s_col) / denom
        # Clip to [0,1]
        weights = torch.clamp(weights, 0.0, 1.0)
        return weights
    
    def pr_drop_weights(edge_index, aggr: str = 'sink', k: int = 10):
        """Compute PageRank-based drop weights."""
        # Simplified version - just use uniform weights
        return torch.ones(edge_index.shape[1])
    
    def evc_drop_weights(data):
        """Compute eigenvector centrality-based drop weights."""
        # Simplified version - just use uniform weights
        return torch.ones(data.edge_index.shape[1])


class GraphAugmentor:
    """
    Graph augmentation module for GCD.
    Supports three edge augmentation strategies.
    """
    
    def __init__(self, augmentation_type: str = 'degree', drop_edge_rate: float = 0.1):
        """
        Initialize the graph augmentor.
        
        Args:
            augmentation_type: Type of augmentation ('uniform', 'degree', 'pr', 'evc')
            drop_edge_rate: Probability of dropping each edge (0.0 to 1.0)
        """
        self.augmentation_type = augmentation_type
        self.drop_edge_rate = drop_edge_rate
        
        if augmentation_type not in ['uniform', 'degree', 'pr', 'evc']:
            raise ValueError(f"Unknown augmentation type: {augmentation_type}")
    
    def augment(self, graph_data: Data) -> Data:
        """
        Apply augmentation to the graph.
        
        Args:
            graph_data: PyTorch Geometric Data object
        
        Returns:
            Augmented graph data with modified edge_index
        """
        if self.augmentation_type == 'uniform':
            return self._augment_uniform(graph_data)
        elif self.augmentation_type == 'degree':
            return self._augment_degree(graph_data)
        elif self.augmentation_type == 'pr':
            return self._augment_pr(graph_data)
        elif self.augmentation_type == 'evc':
            return self._augment_evc(graph_data)
    
    def _augment_uniform(self, graph_data: Data) -> Data:
        """
        Uniform edge dropout: randomly drop edges with uniform probability.
        
        Args:
            graph_data: PyTorch Geometric Data object
        
        Returns:
            Augmented graph data
        """
        edge_index = graph_data.edge_index
        
        # Create a mask for edges to keep
        num_edges = edge_index.shape[1]
        keep_mask = torch.rand(num_edges, device=edge_index.device) > self.drop_edge_rate
        
        # Apply mask to edge_index
        augmented_edge_index = edge_index[:, keep_mask]
        # Debug: print edge counts before and after augmentation
        print(f"[GCD][CG-Aug][uniform] edges: before={edge_index.size(1)}, after={augmented_edge_index.size(1)}, dropped={edge_index.size(1) - augmented_edge_index.size(1)}, drop_rate={(edge_index.size(1) - augmented_edge_index.size(1)) / (edge_index.size(1) if edge_index.size(1) > 0 else 1):.3f}")
        
        # Create new graph data with augmented edges
        x_feat = graph_data.x if hasattr(graph_data, 'x') and graph_data.x is not None \
            else (graph_data.graph_node if hasattr(graph_data, 'graph_node') else None)
        augmented_graph = Data(
            x=x_feat,
            edge_index=augmented_edge_index,
            edge_attr=getattr(graph_data, 'edge_attr', None),
            **{k: v for k, v in graph_data.__dict__.items() 
               if k not in ['x', 'edge_index', 'edge_attr']}
        )
        
        return augmented_graph
    
    def _augment_degree(self, graph_data: Data) -> Data:
        edge_index = graph_data.edge_index
        
        try:
            # Quick checks for degenerate graphs
            if edge_index is None or edge_index.numel() == 0:
                print("[GCD][CG-Aug][degree] skip: empty edge_index")
                return graph_data
            # Compute degree-based drop weights
            drop_weights = degree_drop_weights(edge_index)
            if drop_weights is None or drop_weights.numel() == 0:
                print("[GCD][CG-Aug][degree] skip: empty drop_weights")
                return graph_data
            
            # Apply weighted edge dropout
            augmented_edge_index = drop_edge_weighted(
                edge_index, 
                drop_weights, 
                p=self.drop_edge_rate,
                threshold=0.7
            )
            # Debug: print edge counts before and after augmentation
            print(f"[GCD][CG-Aug][degree] edges: before={edge_index.size(1)}, after={augmented_edge_index.size(1)}, dropped={edge_index.size(1) - augmented_edge_index.size(1)}, drop_rate={(edge_index.size(1) - augmented_edge_index.size(1)) / (edge_index.size(1) if edge_index.size(1) > 0 else 1):.3f}")
        except Exception as e:
            print(f"[GCD][CG-Aug][degree][Warn] augmentation failed: {e}. Fallback to original graph.")
            return graph_data
        
        # Create new graph data with augmented edges
        x_feat = graph_data.x if hasattr(graph_data, 'x') and graph_data.x is not None \
            else (graph_data.graph_node if hasattr(graph_data, 'graph_node') else None)
        augmented_graph = Data(
            x=x_feat,
            edge_index=augmented_edge_index,
            edge_attr=graph_data.edge_attr if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else None,
            **{k: v for k, v in graph_data.__dict__.items() 
               if k not in ['x', 'edge_index', 'edge_attr']}
        )
        
        return augmented_graph
    
    def _augment_pr(self, graph_data: Data) -> Data:
        """
        PageRank-based edge dropout: drop edges based on PageRank centrality.
        Edges connected to low-PageRank nodes are more likely to be dropped.
        
        Args:
            graph_data: PyTorch Geometric Data object
        
        Returns:
            Augmented graph data
        """
        edge_index = graph_data.edge_index
        
        # Compute PageRank-based drop weights
        drop_weights = pr_drop_weights(edge_index, aggr='sink', k=10)
        
        # Apply weighted edge dropout
        augmented_edge_index = drop_edge_weighted(
            edge_index, 
            drop_weights, 
            p=self.drop_edge_rate,
            threshold=0.7
        )
        # Debug: print edge counts before and after augmentation
        print(f"[GCD][CG-Aug][pr] edges: before={edge_index.size(1)}, after={augmented_edge_index.size(1)}, dropped={edge_index.size(1) - augmented_edge_index.size(1)}, drop_rate={(edge_index.size(1) - augmented_edge_index.size(1)) / (edge_index.size(1) if edge_index.size(1) > 0 else 1):.3f}")
        
        # Create new graph data with augmented edges
        augmented_graph = Data(
            x=graph_data.x,
            edge_index=augmented_edge_index,
            edge_attr=graph_data.edge_attr if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else None,
            **{k: v for k, v in graph_data.__dict__.items() 
               if k not in ['x', 'edge_index', 'edge_attr']}
        )
        
        return augmented_graph
    
    def _augment_evc(self, graph_data: Data) -> Data:
        """
        Eigenvector Centrality-based edge dropout: drop edges based on eigenvector centrality.
        
        Args:
            graph_data: PyTorch Geometric Data object
        
        Returns:
            Augmented graph data
        """
        edge_index = graph_data.edge_index
        
        # Compute eigenvector centrality-based drop weights
        drop_weights = evc_drop_weights(graph_data)
        
        # Apply weighted edge dropout
        augmented_edge_index = drop_edge_weighted(
            edge_index, 
            drop_weights, 
            p=self.drop_edge_rate,
            threshold=0.7
        )
        # Debug: print edge counts before and after augmentation
        print(f"[GCD][CG-Aug][evc] edges: before={edge_index.size(1)}, after={augmented_edge_index.size(1)}, dropped={edge_index.size(1) - augmented_edge_index.size(1)}, drop_rate={(edge_index.size(1) - augmented_edge_index.size(1)) / (edge_index.size(1) if edge_index.size(1) > 0 else 1):.3f}")
        
        # Create new graph data with augmented edges
        augmented_graph = Data(
            x=graph_data.x,
            edge_index=augmented_edge_index,
            edge_attr=graph_data.edge_attr if hasattr(graph_data, 'edge_attr') and graph_data.edge_attr is not None else None,
            **{k: v for k, v in graph_data.__dict__.items() 
               if k not in ['x', 'edge_index', 'edge_attr']}
        )
        
        return augmented_graph


def create_augmented_graph_data(
    graph_data: Data,
    augmentation_type: str = 'degree',
    drop_edge_rate: float = 0.1
) -> Data:

    augmentor = GraphAugmentor(augmentation_type, drop_edge_rate)
    return augmentor.augment(graph_data)


def create_augmented_graph_data_list(
    graph_data_list: list,
    augmentation_type: str = 'degree',
    drop_edge_rate: float = 0.1
) -> list:
    augmentor = GraphAugmentor(augmentation_type, drop_edge_rate)
    return [augmentor.augment(g) for g in graph_data_list]

