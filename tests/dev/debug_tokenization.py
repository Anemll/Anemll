#!/usr/bin/env python3
"""
Debug tokenization differences between MLX and ANE evaluation.
Check what special tokens are actually being added.
"""

from transformers import AutoTokenizer

def main():
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Test context from BoolQ
    context = """Ethanol fuel -- All biomass goes through at least some of these steps: it needs to be grown, collected, dried, fermented, distilled, and burned. All of these steps require resources and an infrastructure. The total amount of energy input into the process compared to the energy released by burning the resulting ethanol fuel is known as the energy balance (or ``energy returned on energy invested''). Figures compiled in a 2007 report by National Geographic Magazine point to modest results for corn ethanol produced in the US: one unit of fossil-fuel energy is required to create 1.3 energy units from the resulting ethanol. The energy balance for sugarcane ethanol produced in Brazil is more favorable, with one unit of fossil-fuel energy required to create 8 from the ethanol. Energy balance estimates are not easily produced, thus numerous such reports have been generated that are contradictory. For instance, a separate survey reports that production of ethanol from sugarcane, which requires a tropical climate to grow productively, returns from 8 to 9 units of energy for each unit expended, as compared to corn, which only returns about 1.34 units of fuel energy for each unit of energy expended. A 2006 University of California Berkeley study, after analyzing six separate studies, concluded that producing ethanol from corn uses much less petroleum than producing gasoline.
Question: does ethanol take more energy make that produces?
Answer:"""
    
    print(f"Model: {model_name}")
    print(f"Tokenizer special tokens:")
    print(f"  BOS token: {repr(tokenizer.bos_token)} (ID: {tokenizer.bos_token_id})")
    print(f"  EOS token: {repr(tokenizer.eos_token)} (ID: {tokenizer.eos_token_id})")
    print(f"  PAD token: {repr(tokenizer.pad_token)} (ID: {tokenizer.pad_token_id})")
    print(f"  UNK token: {repr(tokenizer.unk_token)} (ID: {tokenizer.unk_token_id})")
    
    print(f"\nTokenization comparison:")
    
    # Test with add_special_tokens=False
    tokens_false = tokenizer.encode(context, add_special_tokens=False)
    decoded_false = tokenizer.decode(tokens_false)
    
    # Test with add_special_tokens=True  
    tokens_true = tokenizer.encode(context, add_special_tokens=True)
    decoded_true = tokenizer.decode(tokens_true)
    
    print(f"\nWith add_special_tokens=False:")
    print(f"  Length: {len(tokens_false)}")
    print(f"  First 5 tokens: {tokens_false[:5]}")
    print(f"  Last 5 tokens: {tokens_false[-5:]}")
    print(f"  Decoded first 50 chars: {repr(decoded_false[:50])}")
    
    print(f"\nWith add_special_tokens=True:")
    print(f"  Length: {len(tokens_true)}")
    print(f"  First 5 tokens: {tokens_true[:5]}")
    print(f"  Last 5 tokens: {tokens_true[-5:]}")
    print(f"  Decoded first 50 chars: {repr(decoded_true[:50])}")
    
    # Show difference
    if len(tokens_true) != len(tokens_false):
        print(f"\nDifference: {len(tokens_true) - len(tokens_false)} tokens added")
        if len(tokens_true) > len(tokens_false):
            added_tokens = tokens_true[:len(tokens_true)-len(tokens_false)]
            print(f"Added tokens at start: {added_tokens}")
            for token_id in added_tokens:
                print(f"  Token {token_id}: {repr(tokenizer.decode([token_id]))}")
    
    # Test continuations
    print(f"\nContinuation tokenization:")
    continuations = [" no", " yes"]
    for cont in continuations:
        tokens_false = tokenizer.encode(cont, add_special_tokens=False)
        tokens_true = tokenizer.encode(cont, add_special_tokens=True)
        print(f"  {repr(cont)}:")
        print(f"    False: {tokens_false}")
        print(f"    True: {tokens_true}")

if __name__ == "__main__":
    main()