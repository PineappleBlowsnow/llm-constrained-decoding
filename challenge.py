from typing import Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch.nn.functional as F

# --- EXERCISE 1: La disparition (No 'e' or 'E) ---
class LaDisparition:
    """
    Generate text without ever using the letter 'e' or 'E'.
    For this, you must use model() directly: model(input_ids) yields logits.
    You need to manually adjust the logits to forbid tokens containing 'e' or 'E'.
    REQUIREMENT: Do NOT use model.generate().
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        tokenizer_vocab = tokenizer.get_vocab()
        forbidden_token_ids = [token_id for token, token_id in tokenizer_vocab.items() if 'e' in token.lower()]
        self.forbidden_token_ids = torch.tensor(forbidden_token_ids, device=model.device)
        # Here you want to pre-calculate forbidden token IDs

        # Warning: The evaluation server uses a different model and tokenizer than the template. Do not hard-code Token IDs. Use self.tokenizer.get_vocab() or self.tokenizer.encode() to find the IDs relevant to the current model.

    def __call__(self, prompt, max_tokens=30):
        # Tokenize input prompt:
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        
        #Initialize Beam
        beams = [(0.0, input_ids,)]  # Each beam is a tuple of (input_ids, cumulative_log_prob)
        
        for _ in range(max_tokens):
            all_candidates = []
            for cumulative_log_prob, input_ids in beams:
                with torch.no_grad():
                    # Get model outputs
                    outputs = self.model(input_ids=input_ids)
                    logits = outputs.logits 
                    next_token_logits = logits[:, -1, :] 

                # Set the logits of forbidden tokens to a very low value
                next_token_logits[:, self.forbidden_token_ids] = -float('Inf')

                next_token_log_probs = F.log_softmax(next_token_logits, dim=-1)
                topk_log_probs, topk_indices = torch.topk(next_token_log_probs, k=3, dim=-1)

                for i in range(topk_indices.size(1)):
                    token_id = topk_indices[0, i].unsqueeze(0).unsqueeze(0)
                    log_prob = topk_log_probs[0, i].item()
                    new_input_ids = torch.cat([input_ids, token_id], dim=-1)
                    new_cumulative_log_prob = cumulative_log_prob + log_prob
                    all_candidates.append((new_cumulative_log_prob, new_input_ids))

            ordered = sorted(all_candidates, key=lambda x: x[0], reverse=True)
            beams = ordered[:3]  # Keep top 3 beams
            # if beams[0][1][0, -1].item() == self.tokenizer.eos_token_id:
            #     break
        best_score, best_input_ids = beams[0]
   
        return self.tokenizer.decode(best_input_ids[0] , skip_special_tokens=True)
      
      # Generate tokens manually, one step at a time:
      # (The bulk of the logic goes here)
      # Hint: generating a single answer may not be enough!
  
      # Decode output tokens to string and return                   
      # return tokenizer.decode(generated, skip_special_tokens=True)
      #pass



# --- EXERCISE 2: The Toulouse Sequence ---
class ToulouseSequence:
    """
    Generate text without ever using the word 'Toulouse'.
    For this, you must use model() directly: model(input_ids) yields logits.
    You need to manually adjust the logits. It is more difficult here because
    'Toulouse' is a multi-token word.
    REQUIREMENT: Do NOT use model.generate().
    """
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        
        variants = self.get_variant()  
        
    
        self.forbidden_single_ids = set()
        self.forbidden_sequences = []

        for part1, part2 in variants:
    
            ids1 = tokenizer.encode(part1, add_special_tokens=False)
            ids2 = tokenizer.encode(part2, add_special_tokens=False)
            ids = ids1 + ids2
            if len(ids) == 0:
                continue
            if len(ids) == 1:
                self.forbidden_single_ids.add(ids[0])
            else:
        
                self.forbidden_sequences.append((ids[:-1], ids[-1]))
            

    def get_variant(self):
       
        base_words = ["Toulouse", "toulouse", "TOULOUSE"]
        


        sp = ['', ' ', '-', '\n', '\b','\f','\r','\t','\v']
        result = []
        for first in sp:
            for second in base_words:
                for third in sp:
                    variant = first + second + third
                    result.append(variant)
        final_res = set()
        for word in result:
            for i in range(0, len(word)):
                final_res.add((word[:i],word[i:]))
        return final_res
    

    def __call__(self, prompt, max_tokens=30):
        input_ids = self.tokenizer.encode(prompt, return_tensors="pt").to(self.model.device)
        len_prompt = input_ids.shape[1]
        beam_width = max([len(s) for s in self.forbidden_sequences])
        beams = [(0.0, input_ids)]
        
        for _ in range(max_tokens):
            candidates = []
            
            for score, seq in beams:
                if seq[0, -1] == self.tokenizer.eos_token_id: # shape of seq is (1, seq_len)
                    candidates.append((score, seq))
                    continue

                with torch.no_grad():
                    outputs = self.model(seq)
                    next_token_logits = outputs.logits[:, -1, :] 

                if self.forbidden_single_ids:
                    next_token_logits[:, list(self.forbidden_single_ids)] = -float('inf')

                for prefix, target_id in self.forbidden_sequences:
                    prefix_len = len(prefix)
                    if seq.shape[1] >= prefix_len:
                        current_suffix = seq[0, -prefix_len:].tolist()
                        if current_suffix == prefix:
                            next_token_logits[:, target_id] = -float('inf')

                next_log_probs = F.log_softmax(next_token_logits, dim=-1)
                top_scores, top_ids = torch.topk(next_log_probs, beam_width)
                
                for i in range(beam_width):
                    token_id = top_ids[0, i].unsqueeze(0).unsqueeze(0)
                    token_score = top_scores[0, i].item()
                    
                    new_score = score + token_score
                    new_seq = torch.cat([seq, token_id], dim=-1)
                    
                    candidates.append((new_score, new_seq))

            ordered = sorted(candidates, key=lambda x: x[0], reverse=True)
            beams = ordered[:beam_width]

            if beams[0][1][0, -1] == self.tokenizer.eos_token_id:
                break
        
        best_seq = beams[0][1]
        return self.tokenizer.decode(best_seq[0,len_prompt:], skip_special_tokens=True)
        # Generate tokens manually, one step at a time:
        # (The bulk of the logic goes here)
        # Hint: you need to track partial matches of the forbidden word
    
        # Decode output tokens to string and return                   
        # return tokenizer.decode(generated, skip_special_tokens=True)
        # pass
    
if __name__ == "__main__":
    MODEL_NAME = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, dtype=torch.float16, device_map="auto")
    la_disparition_generator = LaDisparition(model, tokenizer)
    print("Ex 1 (No 'e'):", la_disparition_generator("Describe a cat."))
    toulouse_sequence_generator = ToulouseSequence(model, tokenizer)
    print("Ex 2 (No 'Toulouse'):", toulouse_sequence_generator("The pink city in France is"))
