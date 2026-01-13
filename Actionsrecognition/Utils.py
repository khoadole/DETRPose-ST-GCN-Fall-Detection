"""
Graph utilities for ST-GCN with COCO-17 keypoints support.

Modified from Human-Falling-Detect-Tracks to support 17 COCO keypoints
instead of 14 (COCO-cut).

Reference: https://github.com/yysijie/st-gcn/blob/master/net/utils/graph.py
"""

import os
import torch
import numpy as np


class Graph:
    """
    The Graph to model the skeletons extracted by pose estimation models.
    
    Args:
        - strategy: (string) must be one of:
            - uniform: Uniform Labeling
            - distance: Distance Partitioning
            - spatial: Spatial Configuration
        - layout: (string) must be one of:
            - coco_cut: COCO format with 4 joints cut (14 nodes) - original
            - coco17: Full COCO 17 keypoints
        - max_hop: (int) maximal distance between two connected nodes
        - dilation: (int) controls spacing between kernel points
    """
    
    def __init__(self,
                 layout='coco17',
                 strategy='spatial',
                 max_hop=1,
                 dilation=1):
        self.max_hop = max_hop
        self.dilation = dilation
        
        self.get_edge(layout)
        self.hop_dis = get_hop_distance(self.num_node, self.edge, max_hop)
        self.get_adjacency(strategy)
    
    def get_edge(self, layout):
        """Define skeleton graph edges based on layout."""
        
        if layout == 'coco_cut':
            # Original 14-node layout (eyes and ears removed)
            self.num_node = 14
            self_link = [(i, i) for i in range(self.num_node)]
            neighbor_link = [
                (6, 4), (4, 2), (2, 13), (13, 1), (5, 3), (3, 1),
                (12, 10), (10, 8), (8, 2), (11, 9), (9, 7), (7, 1), (13, 0)
            ]
            self.edge = self_link + neighbor_link
            self.center = 13  # Center point (added)
            
        elif layout == 'coco17':
            # Full COCO 17 keypoints
            # Index mapping:
            # 0: Nose
            # 1: Left Eye      2: Right Eye
            # 3: Left Ear      4: Right Ear
            # 5: Left Shoulder 6: Right Shoulder
            # 7: Left Elbow    8: Right Elbow
            # 9: Left Wrist    10: Right Wrist
            # 11: Left Hip     12: Right Hip
            # 13: Left Knee    14: Right Knee
            # 15: Left Ankle   16: Right Ankle
            
            self.num_node = 17
            self_link = [(i, i) for i in range(self.num_node)]
            
            # COCO skeleton connections
            neighbor_link = [
                # Face connections
                (0, 1), (0, 2),      # Nose to Eyes
                (1, 3), (2, 4),      # Eyes to Ears
                
                # Upper body
                (5, 6),              # Shoulder to Shoulder
                (5, 7), (7, 9),      # Left Arm: Shoulder -> Elbow -> Wrist
                (6, 8), (8, 10),     # Right Arm: Shoulder -> Elbow -> Wrist
                
                # Torso
                (5, 11), (6, 12),    # Shoulders to Hips
                (11, 12),            # Hip to Hip
                
                # Lower body
                (11, 13), (13, 15),  # Left Leg: Hip -> Knee -> Ankle
                (12, 14), (14, 16),  # Right Leg: Hip -> Knee -> Ankle
                
                # Ear to Shoulder (for head orientation)
                (3, 5), (4, 6),      # Ears to Shoulders
            ]
            
            self.edge = self_link + neighbor_link
            self.center = 0  # Nose as center
            
        else:
            raise ValueError(f'Layout "{layout}" is not supported!')
    
    def get_adjacency(self, strategy):
        """Calculate adjacency matrix based on partitioning strategy."""
        valid_hop = range(0, self.max_hop + 1, self.dilation)
        adjacency = np.zeros((self.num_node, self.num_node))
        
        for hop in valid_hop:
            adjacency[self.hop_dis == hop] = 1
        normalize_adjacency = normalize_digraph(adjacency)
        
        if strategy == 'uniform':
            A = np.zeros((1, self.num_node, self.num_node))
            A[0] = normalize_adjacency
            self.A = A
            
        elif strategy == 'distance':
            A = np.zeros((len(valid_hop), self.num_node, self.num_node))
            for i, hop in enumerate(valid_hop):
                A[i][self.hop_dis == hop] = normalize_adjacency[self.hop_dis == hop]
            self.A = A
            
        elif strategy == 'spatial':
            A = []
            for hop in valid_hop:
                a_root = np.zeros((self.num_node, self.num_node))
                a_close = np.zeros((self.num_node, self.num_node))
                a_further = np.zeros((self.num_node, self.num_node))
                
                for i in range(self.num_node):
                    for j in range(self.num_node):
                        if self.hop_dis[j, i] == hop:
                            if self.hop_dis[j, self.center] == self.hop_dis[i, self.center]:
                                a_root[j, i] = normalize_adjacency[j, i]
                            elif self.hop_dis[j, self.center] > self.hop_dis[i, self.center]:
                                a_close[j, i] = normalize_adjacency[j, i]
                            else:
                                a_further[j, i] = normalize_adjacency[j, i]
                
                if hop == 0:
                    A.append(a_root)
                else:
                    A.append(a_root + a_close)
                    A.append(a_further)
            
            A = np.stack(A)
            self.A = A
            
        else:
            raise ValueError(f'Strategy "{strategy}" is not supported!')


def get_hop_distance(num_node, edge, max_hop=1):
    """Calculate hop distance between nodes."""
    A = np.zeros((num_node, num_node))
    for i, j in edge:
        A[j, i] = 1
        A[i, j] = 1
    
    # Compute hop steps
    hop_dis = np.zeros((num_node, num_node)) + np.inf
    transfer_mat = [np.linalg.matrix_power(A, d) for d in range(max_hop + 1)]
    arrive_mat = (np.stack(transfer_mat) > 0)
    
    for d in range(max_hop, -1, -1):
        hop_dis[arrive_mat[d]] = d
    
    return hop_dis


def normalize_digraph(A):
    """Normalize directed graph adjacency matrix."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-1)
    
    AD = np.dot(A, Dn)
    return AD


def normalize_undigraph(A):
    """Normalize undirected graph adjacency matrix."""
    Dl = np.sum(A, 0)
    num_node = A.shape[0]
    Dn = np.zeros((num_node, num_node))
    
    for i in range(num_node):
        if Dl[i] > 0:
            Dn[i, i] = Dl[i] ** (-0.5)
    
    DAD = np.dot(np.dot(Dn, A), Dn)
    return DAD


# Test the graph
if __name__ == "__main__":
    print("Testing COCO-17 Graph...")
    
    graph = Graph(layout='coco17', strategy='spatial')
    
    print(f"Number of nodes: {graph.num_node}")
    print(f"Number of edges: {len(graph.edge)}")
    print(f"Adjacency matrix shape: {graph.A.shape}")
    print(f"Center node: {graph.center}")
    
    print("\nEdge list:")
    for edge in graph.edge:
        if edge[0] != edge[1]:  # Skip self-loops
            print(f"  {edge}")
