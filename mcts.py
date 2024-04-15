import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import time
from tqdm import tqdm

class mcts():
    '''
    A token sampling class where it uses AlphaGo style MCTS without v(x, a) but only p(x, a).
    '''
    def __init__(self, model, depth, batch, nodes: int = 1024**2, actions: int = 32):
        self.max_nodes = nodes
        self.actions = actions
        self.max_depth = depth
        self.max_batch = batch
        self.model = model
    
    def initialize_nodes(self):
        self.curr = 0
        self.Q = torch.zeros(self.max_nodes, dtype = torch.float32)
        self.P = torch.zeros((self.max_nodes, self.actions), dtype = torch.bfloat16)
        self.N = torch.zeros((self.max_nodes, self.actions), dtype = torch.uint16)
        self.next = torch.zeros((self.max_nodes, self.actions), dtype = torch.uint32)
        self.prev = torch.zeros(self.max_nodes, dtype = torch.uint32)
    
    def add_nodes(self, curr_ptrs, new_tokens):
        '''
        curr_ptrs: torch.uint16[<max_batch] # ptrs can be the same node
        new_tokens: torch.uint16[<max_batch]
        '''
        assert curr_ptrs.size == new_tokens.size
        

    def expand(self, curr_ptr: int, batch_size: int, codestr: str, depth: int, max_depth: int):
        '''
        Simultaneously samples node & expand leaf nodes batch_size times recursively until terminal state.
        There are two modes: traversal and expand.
        During traversal we are going through existing nodes, expand will add node.
        '''
        logits = self.model(results[:, num_new_tokens-1 : num_new_tokens], curr_pos)
        
    def search(self, prompt: str, root_node: int, batch_size: int, max_depth: int):
        '''continuously expand at root node using batch_size until a solution is found'''
        self.model.resetTimer()
        curr_pos = 0
        # forward the model to get the logits for the index in the sequence
        if (curr_pos == 0):
            logits = self.model(idx_cond[:, curr_pos:], curr_pos)
            curr_pos += idx_cond.shape[1]
        else:
            
        
    
    def eval(self, codestr: str):
        '''eval code to get a reward'''

