import math
from dataclasses import dataclass
from typing import Any, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import nn
import time
from tqdm import tqdm

# 518 appears if you have indentation
TERMINAL_SEQ = torch.tensor([    29961, 29914, 20055,  4690,  1164, 29962], device="cuda:0")

def detectTerminal(token: torch.Tensor, state: torch.Tensor):
    """
    Terminates if detected [/PYTHON] or </end>
    Input: token[batch_size], state[batch_size]
    Output: isEnd[batch_size], state[batch_size]
    """
    mask = (torch.gather(TERMINAL_SEQ, 0, state.to(torch.int64))==token).int()
    state = (state + mask) * mask
    end = (state == len(TERMINAL_SEQ)) | (token==2)
    state = (~end).int() * state
    #if (torch.sum(end) > 0):
        #print(state, mask, token)
        #breakpoint()
        #print("Early termination", torch.sum(end))
    return end, state

class mcts():
    '''
    A token sampling class where it uses AlphaGo style MCTS without v(x, a) but only p(x, a).
    '''
    def __init__(self, model, depth, nodes: int = 1024**2, top_k: int = 32, temp: int = 0.3):
        self.max_nodes = nodes
        self.top_k = top_k
        self.max_depth = depth
        self.max_batch = model.params.max_batch_size
        self.model = model
        self.temp = temp
        self.initialize_nodes()
    
    def initialize_nodes(self):
        self.curr = 0
        self.Q = torch.zeros(self.max_nodes, dtype = torch.float32)
        self.P = torch.zeros((self.max_nodes, self.top_k), dtype = torch.bfloat16)
        self.N = torch.zeros((self.max_nodes, self.top_k), dtype = torch.int32)
        self.next_ptr = torch.zeros((self.max_nodes, self.top_k), dtype = torch.int32)
        self.prev_ptr = torch.zeros(self.max_nodes, dtype = torch.int32)
    
    def add_nodes(self, curr_ptrs, new_tokens):
        '''
        curr_ptrs: torch.uint16[<max_batch] # ptrs can be the same node
        new_tokens: torch.uint16[<max_batch]
        '''
        #assert curr_ptrs.size == new_tokens.size
        

    def expand(self, curr_ptr: int, batch_size: int, codestr: str, depth: int, max_depth: int):
        '''
        Simultaneously samples node & expand leaf nodes batch_size times recursively until terminal state.
        There are two modes: traversal and expand.
        During traversal we are going through existing nodes, expand will add node.
        '''
        #logits = self.model(results[:, num_new_tokens-1 : num_new_tokens], curr_pos)
        
    def search(self, prompt: torch.Tensor, batch_size: int):
        '''continuously expand at root node using batch_size until a solution is found'''
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        """
        self.model.resetTimers()
        self.model.resetSeq()

        print("Decoding with: batch size =", batch_size)
        assert self.model.params.max_batch_size >= batch_size
        idx = prompt.expand(1, -1)
        print("idx.shape", idx.shape)
        curr_pos = 0
        assert self.model.params.max_prompt_seq_len >= idx.shape[1]
        max_new_tokens = min(self.max_depth, self.model.params.max_seq_len)
        results = torch.zeros(batch_size, max_new_tokens, dtype=torch.int64, device = idx.device)
        results_len = torch.zeros(batch_size, dtype=torch.int32, device = idx.device)
        results_mask = results_len == 0
        print("results.shape", results.shape)
        curr_batch_size = 1
        output_counts = torch.zeros((2, batch_size), dtype=torch.int32, device = idx.device)
        output_counts[1, 0] = batch_size

        batch_stats = torch.zeros((max_new_tokens), dtype=torch.uint8, device = idx.device)
        end_stateMachine = torch.zeros((batch_size), dtype=torch.uint8, device = idx.device)

        for num_new_tokens in tqdm(range(max_new_tokens), desc="Tokens", leave=False):
            # forward the model to get the logits for the index in the sequence
            if (curr_pos == 0):
                # if the sequence context is growing too long we must crop it at block_size
                idx_cond = idx if idx.size(1) <= self.model.params.max_prompt_seq_len else idx[:, -self.model.params.max_prompt_seq_len:]
                logits = self.model(idx_cond[:, curr_pos:], curr_pos)
                curr_pos += idx_cond.shape[1]
            else:
                logits = self.model(results[:curr_batch_size, num_new_tokens-1 : num_new_tokens], curr_pos)
                curr_pos += 1

            logits = logits[:, -1, :] # crop to just the final time step
            #print("logits.shape", logits.shape)

            # DEAL WITH EACH TOKEN'S SAMPLING & BATCH DIVERGENCE PROBLEM
            
            # pluck the logits at the final step and scale by desired temperature
            logits = logits / self.temp
            
            v, top_k_idx = torch.topk(logits, min(self.top_k, logits.size(-1)))
            
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(v, dim=-1)
            
            # sample multiple times
            #print("Output, counts", output_counts)

            for i in range(curr_batch_size):
                sample = torch.multinomial(probs[i], num_samples=output_counts[1, i], replacement=True)
                next_token = torch.gather(top_k_idx[i], 0, sample) # convert to actual token
                #print(next_token)
                unique_idx, counts = torch.unique(next_token, return_counts=True, dim=0)
                #print("unique, counts", unique_idx, counts)
                
                # fork first
                if results_mask[i]:
                    output_counts[0, i] = unique_idx[0]
                    output_counts[1, i] = counts[0]
                    results[i, num_new_tokens] = unique_idx[0]

                # fork rest
                forks = len(counts)-1
                forks = min(forks, batch_size-curr_batch_size)
                if (forks > 0) & (curr_batch_size < batch_size) & results_mask[i]:
                    #breakpoint()
                    self.model.fork(i, curr_batch_size, forks)
                    for j in range(forks):
                        results[curr_batch_size+j] = results[i]
                        output_counts[0, curr_batch_size+j] = unique_idx[1+j]
                        output_counts[1, curr_batch_size+j] = counts[1+j]
                        results[curr_batch_size+j, num_new_tokens] = unique_idx[1+j]

                    curr_batch_size += forks
            
            assert curr_batch_size <= batch_size, f"Batch size mismatch {curr_batch_size}!={batch_size}"
            assert torch.sum(output_counts[1]) == batch_size, f"Batch size mismatch {torch.sum(counts)}!={batch_size}"

            # append sampled index to the running sequence and continue
            # idx = torch.cat((idx, idx_next), dim=1)
            results[:, num_new_tokens] *= results_mask.int()
            #print(idx_next[:, 0])

            ends, end_stateMachine = detectTerminal(results[:, num_new_tokens], end_stateMachine)
            results_len += results_mask.int()
            #print(results_mask.int(), results_len, (~ends).int() * 1)
            results_mask = ~((ends) | ~results_mask)
            #breakpoint()
            batch_stats[num_new_tokens] = curr_batch_size

            #if enc is not None:
                #print(enc(idx, skip_special_tokens = True)[0])
                #print(enc.decode(idx[0].tolist()))
            
            if (torch.sum(~results_mask) == curr_batch_size):
                self.model.printTimers()
                batch_stats = batch_stats[:num_new_tokens+1]
                print(batch_stats)
                print("Early stopping...")
                return results[:curr_batch_size, :num_new_tokens+1], results_len, batch_stats
        
        self.model.printTimers()
        print(batch_stats)
        return results[:curr_batch_size, :], results_len, batch_stats
    
    def eval(self, codestr: str):
        '''eval code to get a reward'''
