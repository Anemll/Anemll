#  Copyright (c) 2025, Anemll  All rights reserved.
#
#  Use of this source code is governed by a MIT license that can be
#  found in the LICENSE.txt file or at https://opensource.org/license/mit

import unittest
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
import os
import time
import pytest


#  TEST with BATCHED PREFILL
# python -m unittest tests.test_py_llama.TestLlamaModel.test_generation_with_prefill -v
#
#  TEST with SINGLE PREFILL
# python -m unittest tests.test_py_llama.TestLlamaModel.test_generation_with_single_prefill -v
#  MODEL_PATH is location for HF model

MODEL_PATH = os.path.expanduser("../Meta-Llama-3.2-1B")

# Update imports to use package imports
from anemll.models.llama_model import (  # noqa: E402
    LlamaConfig,
    LlamaForCausalLM,
    MODEL_DTYPE,
    TEST_DEVICE,  # Import TEST_DEVICE from llama_model.py
    CONTEXT_LENGTH,
    STATE_LENGTH,
    ENABLE_CONV2D,
    ENABLE_VACAB_SPLIT,
    ENABLE_VACAB_SPLIT8,
    ENABLE_LOGITS2,
    ENABLE_DEBUG3,
)

# Skip the entire module if local weights are unavailable
pytest.skip("Requires local Llama weights", allow_module_level=True)

print(f"Using imported TEST_DEVICE: {TEST_DEVICE}")


def make_causal_mask(length, start):
    """Create a causal mask for attention to prevent tokens from attending to future positions.

    The mask ensures each token can only attend to previous tokens and itself by setting future
    positions to negative infinity in the attention weights.

    Args:
        length (int): The sequence length for the mask
        start (int): Starting position for causal masking

    Returns:
        torch.Tensor: Causal mask of shape (1, 1, length, length) on TEST_DEVICE, where valid
                     attention positions are 0 and invalid (future) positions are -inf
    """
    # Initialize the mask with -inf on correct device and dtype
    min_val = torch.finfo(MODEL_DTYPE).min
    mask = torch.full(
        (1, 1, length, length), min_val, dtype=MODEL_DTYPE, device=TEST_DEVICE
    )

    # Create mask condition based on positions - ensure on correct device
    mask_cond = torch.arange(length, device=TEST_DEVICE)
    mask_cond = mask_cond < (mask_cond + 1).view(length, 1)

    # Fill in allowed positions with 0
    mask[0, 0, mask_cond] = 0

    return mask


def initialize_tokenizer(model_path):
    """Initialize and configure the tokenizer for the Llama model.

    This function loads the tokenizer from the specified model path and configures it with
    appropriate padding tokens and settings. It handles cases where pad tokens may not be
    defined in the original tokenizer.

    Args:
        model_path (str): Path to the model directory containing tokenizer files

    Returns:
        AutoTokenizer: Configured tokenizer instance with proper padding settings

    Raises:
        Exception: If tokenizer loading fails, returns None and prints error
    """
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
        print("Tokenizer loaded successfully using AutoTokenizer.")
        print(f"Tokenizer vocabulary size: {len(tokenizer)}")

        print(f"Tokenizer default padding side: {tokenizer.padding_side}")
        tokenizer.padding_side = "left"
        print(f"Tokenizer new padding side: {tokenizer.padding_side}")

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id
            print("Pad token was not defined. Set to EOS token.")

        # Set pad token to "space" if it's not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = " "
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(" ")
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token_id = 0  # Use 0 as a fallback
            print("Pad token was not defined. Set to space character.")

        print(f"Tokenizer pad token: {tokenizer.pad_token}")
        print(f"Tokenizer pad token ID: {tokenizer.pad_token_id}")

        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None


class TestLlamaModel(unittest.TestCase):
    """Test suite for the Llama model implementation.

    This test suite verifies various aspects of the Llama model including:
    - Model initialization and configuration
    - Attention mechanism functionality
    - KV cache updates and management
    - Token generation capabilities
    - Prefill mode operation
    - Integration with tokenizer
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures and model initialization.

        This method:
        1. Loads model configuration from json
        2. Initializes the LlamaForCausalLM model
        3. Loads pretrained weights if available
        4. Verifies proper weight loading
        5. Sets model to evaluation mode

        Raises:
            RuntimeError: If config file not found or weights fail to load properly
        """
        print("\nsetUpClass is being called...")
        print(f"Using device: {TEST_DEVICE}")

        # Load config from json file
        config_path = os.path.join(MODEL_PATH, "config.json")
        if os.path.exists(config_path):
            print(f"Loading config from {config_path}")
            cls.config = LlamaConfig.from_json(config_path)
        else:
            print(f"Error: {config_path} not found, using default config")
            raise RuntimeError(f"Error: {config_path} not found, using default config")

        # Set the dtype for tests
        cls.dtype = MODEL_DTYPE

        print("Loaded model config:")
        print(f"  hidden_size: {cls.config.hidden_size}")
        print(f"  intermediate_size: {cls.config.intermediate_size}")
        print(f"  num_attention_heads: {cls.config.num_attention_heads}")
        print(f"  num_hidden_layers: {cls.config.num_hidden_layers}")
        print(f"  num_key_value_heads: {cls.config.num_key_value_heads}")
        print(f"  vocab_size: {cls.config.vocab_size}")
        print(f"  max_position_embeddings: {cls.config.max_position_embeddings}")
        print(f"  rope_theta: {cls.config.rope_theta}")

        # Initialize model
        cls.model = LlamaForCausalLM(
            cls.config, enable_coreml=False, use_ane_norm=False
        )

        print(f"  cls.config.torch_required: {cls.config.torch_required}")

        # Load pretrained weights if available
        if os.path.exists(MODEL_PATH):
            print(f"Loading pretrained weights from {MODEL_PATH}")
            success = cls.model.load_pretrained_weights(  # Call on base model instead
                MODEL_PATH,
                enable_conv2d=ENABLE_CONV2D,
                enable_vocab_split=ENABLE_VACAB_SPLIT,
                enable_vocab_split8=ENABLE_VACAB_SPLIT8,
                enable_logits2=ENABLE_LOGITS2,
                enable_coreml=False,
                mlp_up_split=1,
                mlp_down_split=1,
                enable_debug=ENABLE_DEBUG3,
            )
            if not success:
                raise RuntimeError("Failed to load pretrained weights")

            # Verify weights are loaded
            for name, param in cls.model.named_parameters():
                if torch.all(param == 0):
                    raise RuntimeError(
                        f"Parameter {name} contains all zeros - weights not loaded properly"
                    )
        else:
            print(f"Warning: {MODEL_PATH} not found, using random weights")

        cls.model.eval()

    def test_model_initialization(self):
        """Verify basic model initialization parameters.

        Tests:
        - Hidden size configuration
        - Number of layers
        - Number of attention heads
        - Number of key-value heads
        """
        self.assertEqual(self.model.config.hidden_size, 2048)
        self.assertEqual(self.model.config.num_hidden_layers, 16)
        self.assertEqual(self.model.config.num_attention_heads, 32)
        self.assertEqual(self.model.config.num_key_value_heads, 8)

    def test_attention_shapes(self):
        """Verify attention mechanism tensor shapes and operations.

        Tests:
        - Input tensor shapes for attention computation
        - Forward pass through attention layers
        - Output tensor shapes and dtypes
        - Proper device placement of tensors
        """
        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model = self.model.to(TEST_DEVICE)

        batch_size, seq_length = 1, 1

        # Create dummy input_ids
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, seq_length), device=TEST_DEVICE
        )

        # Create attention inputs - ensure all tensors are on correct device
        update_mask = torch.zeros(
            (batch_size, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )
        position_ids = torch.tensor(
            [0], dtype=torch.long, device=TEST_DEVICE
        )  # Make it 1D
        causal_mask = make_causal_mask(
            seq_length, 0
        )  # This helper function should handle device placement
        current_pos = torch.tensor([0], device=TEST_DEVICE)
        single_causal_mask = causal_mask[
            :, :, current_pos : current_pos + 1, :CONTEXT_LENGTH
        ]

        # Run forward pass through LlamaForCausalLM
        with torch.no_grad():
            output = self.model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=single_causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False,
            )

        # Check output shape and dtype
        # For LlamaForCausalLM, output should be logits with shape [batch_size, seq_length, vocab_size]
        self.assertEqual(output.shape, (batch_size, seq_length, self.config.vocab_size))
        self.assertEqual(output.dtype, MODEL_DTYPE)
        self.assertEqual(
            str(output.device), str(TEST_DEVICE)
        )  # Compare string representations of devices

        # Verify output contains valid values
        self.assertFalse(torch.any(torch.isnan(output)))
        self.assertFalse(torch.any(torch.isinf(output)))

    def test_kv_cache_update(self):
        """Verify key-value cache updates and management.

        Tests:
        - Cache initialization
        - Cache updates during forward pass
        - Cache shape and dtype correctness
        - Non-zero values in cache after updates
        """
        batch_size, seq_length = 1, 1

        # Create dummy input_ids instead of hidden_states
        input_ids = torch.randint(
            0, self.config.vocab_size, (batch_size, seq_length), device=TEST_DEVICE
        )

        # Create cache update inputs - ensure all tensors are on correct device
        update_mask = torch.ones(
            (batch_size, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
        )
        position_ids = torch.tensor(
            [0], dtype=torch.long, device=TEST_DEVICE
        )  # Make it 1D
        causal_mask = torch.zeros(1, 1, seq_length, CONTEXT_LENGTH, device=TEST_DEVICE)
        current_pos = torch.tensor([0], device=TEST_DEVICE)

        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model = self.model.to(TEST_DEVICE)

        # Initial forward pass
        with torch.no_grad():
            _ = self.model(
                input_ids=input_ids,
                update_mask=update_mask,
                position_ids=position_ids,
                causal_mask=causal_mask,
                current_pos=current_pos,
                IN_PREFILL=False,
            )

        # Check cache shapes and values
        cache_name = "kv_cache_0"
        if hasattr(
            self.model.model, cache_name
        ):  # Check model.model since we're using LlamaForCausalLM
            cache = getattr(self.model.model, cache_name)
            expected_shape = (
                2 * self.config.num_hidden_layers,  # 2 for key and value caches
                self.config.num_key_value_heads,
                STATE_LENGTH,
                self.config.hidden_size // self.config.num_attention_heads,
            )
            self.assertEqual(cache.shape, expected_shape)
            self.assertEqual(cache.dtype, MODEL_DTYPE)
            # Check that cache is not all zeros after update
            self.assertFalse(torch.all(cache == 0))

    def test_causal_lm_forward(self):
        """Test the causal language model's forward pass.

        Tests:
        - Input processing
        - Forward pass computation
        - Output logits shape and values
        - Proper probability distribution in output
        """
        model = LlamaForCausalLM(self.config, enable_coreml=False)
        model.eval()

        batch_size, seq_length = 1, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_length))

        # Create inputs
        update_mask = torch.zeros((batch_size, 1, CONTEXT_LENGTH, 1), dtype=self.dtype)
        position_ids = torch.arange(seq_length).unsqueeze(0)
        causal_mask = (
            torch.triu(torch.ones(1, 1, seq_length, seq_length), diagonal=1) * -1e4
        )
        current_pos = torch.tensor([0])

        # Run forward pass
        output = model(
            input_ids=input_ids,
            update_mask=update_mask,
            position_ids=position_ids,
            causal_mask=causal_mask,
            current_pos=current_pos,
        )

        # Check output shape and values
        expected_shape = (batch_size, seq_length, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        self.assertEqual(output.dtype, self.dtype)
        # Check that output contains valid probabilities
        self.assertFalse(torch.any(torch.isnan(output)))
        self.assertFalse(torch.any(torch.isinf(output)))

    def test_rotary_embeddings(self):
        """Test rotary position embeddings computation.

        Tests:
        - Embedding generation for positions
        - Shape and dtype of embeddings
        - Value range constraints
        - Proper device placement
        """
        # Get rotary embeddings
        current_pos = torch.tensor([0])
        cos, sin = self.model.get_rotary_embeddings_s(current_pos)

        # Check shapes and values
        head_dim = self.config.hidden_size // self.config.num_attention_heads
        self.assertEqual(cos.shape, (1, 1, 1, head_dim))
        self.assertEqual(sin.shape, (1, 1, 1, head_dim))
        self.assertEqual(cos.dtype, self.dtype)
        self.assertEqual(sin.dtype, self.dtype)
        # Check that values are in valid range [-1, 1]
        self.assertTrue(torch.all(cos >= -1) and torch.all(cos <= 1))
        self.assertTrue(torch.all(sin >= -1) and torch.all(sin <= 1))

    def test_generation_with_prefill(self):
        """Test complete generation pipeline with batched prefill.

        This test verifies:
        1. Tokenizer initialization and EOT token handling
        2. Prompt processing and padding
        3. Batched prefill of prompt tokens
        4. Token-by-token generation
        5. EOT token detection and generation stopping
        6. Proper text decoding and output formatting
        """
        # Initialize tokenizer
        tokenizer = initialize_tokenizer(MODEL_PATH)
        print("\n[DEBUG] Tokenizer info:")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
        print(f"  Special tokens: {tokenizer.special_tokens_map}")

        # Find the actual EOT token ID from the template
        test_message = [{"role": "system", "content": "test"}]
        test_ids = tokenizer.apply_chat_template(test_message, return_tensors="pt")[0]
        print("\n[DEBUG] Template token analysis:")
        print("Token sequence:")
        for i, token_id in enumerate(test_ids):
            token = tokenizer.decode([token_id])
            print(f"  {i}: {token_id} -> '{token}'")
            if "<|eot_id|>" in token:
                print(f"  Found EOT token ID: {token_id}")
                eot_token_id = token_id

        batch_size = 64

        # Prepare input
        prompt = "What is Apple Neural Engine?"
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )

        decoded_input = tokenizer.decode(input_ids[0])
        print("\n[DEBUG] Decoded content:")
        print(f"  Prompt: {prompt}")
        print(f"  Full decoded input: {decoded_input}")
        print("  Token by token:")
        for i, token_id in enumerate(input_ids[0]):
            token = tokenizer.decode([token_id])
            print(f"    {i}: {token_id} -> '{token}'")

        # Ensure input_ids is on correct device
        input_ids = input_ids.to(TEST_DEVICE)

        prompt_length = input_ids.shape[1]

        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model = self.model.to(TEST_DEVICE)

        # Pad to full context length
        print("\n[DEBUG] Before context padding:")
        print(f"  Current shape: {input_ids.shape}")
        print(f"  Padding to context length: {CONTEXT_LENGTH}")

        input_ids = F.pad(
            input_ids, (0, CONTEXT_LENGTH - prompt_length), value=tokenizer.pad_token_id
        )
        print(f"[DEBUG] input_ids length: {input_ids.shape[1]} ")

        if ENABLE_DEBUG3:
            print("\nInput preparation:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  input_ids device: {input_ids.device}")
            print(f"  input_ids dtype: {input_ids.dtype}")
            print(f"  prompt_length: {prompt_length}")
            print(f"  model device: {next(self.model.parameters()).device}")

        # Start timing prefill phase
        prefill_start = time.time()

        # Prefill phase - process in batches
        print("\n[DEBUG] Starting prefill phase:")
        print(f"  Total prompt length: {prompt_length}")

        # Process all but the last token of the prompt in batches
        tokens_to_process = prompt_length - 1  # Leave one token for first prediction
        current_pos = 0

        print(
            f"  Tokens to process: {tokens_to_process} (leaving last token for prediction)"
        )

        # Create full causal mask for the entire context length
        causal_mask = make_causal_mask(CONTEXT_LENGTH, 0)

        while current_pos < tokens_to_process:
            # Calculate batch end position
            batch_end = min(current_pos + batch_size, tokens_to_process)
            current_batch_size = batch_end - current_pos

            print("\n[DEBUG] Processing batch:")
            print(f"  Current position: {current_pos}")
            print(f"  Batch end: {batch_end}")
            print(f"  Current batch size: {current_batch_size}")

            # Extract current batch of tokens and ensure it's padded to batch_size
            batch_input = input_ids[:, current_pos:batch_end]

            # Always pad to batch_size (64) tokens
            if batch_input.shape[1] < batch_size:
                print(
                    f"  Padding batch from {batch_input.shape[1]} to {batch_size} tokens"
                )
                batch_input = F.pad(
                    batch_input,
                    (0, batch_size - batch_input.shape[1]),
                    value=tokenizer.pad_token_id,
                )

            # Prepare causal mask for this batch - use full batch_size
            multiple_causal_mask = causal_mask[
                :, :, current_pos : current_pos + batch_size, :
            ]

            # Create position IDs for full batch_size
            position_ids = torch.arange(
                current_pos, current_pos + batch_size, device=TEST_DEVICE
            )

            # Get rotary embeddings for the full batch
            # Run prefill for this batch
            with torch.no_grad():
                self.model.prefill_kv_cache(
                    batch_input,
                    position_ids=position_ids,
                    start_pos=torch.tensor([current_pos], device=TEST_DEVICE),
                    causal_mask=multiple_causal_mask,
                )

            current_pos = batch_end

        # Set current_pos for generation phase to the last token position
        current_pos = torch.tensor(
            [tokens_to_process], device=TEST_DEVICE
        )  # Position of last token

        print("\n[DEBUG] Prefill complete:")
        print(f"  Final position: {current_pos.item()}")
        print(f"  Ready for generation starting at position: {current_pos.item()}")

        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        prefill_tokens_per_second = tokens_to_process / prefill_time
        print("\n[TIMING] Prefill phase:")
        print(f"  Tokens processed: {tokens_to_process}")
        print(f"  Time taken: {prefill_time:.2f} seconds")
        print(f"  Speed: {prefill_tokens_per_second:.2f} tokens/second")

        # Start timing inference phase
        inference_start = time.time()
        total_new_tokens = 0

        # Generation phase starts here
        print(f"\n[DEBUG] Starting generation phase from position {current_pos.item()}")

        # Initialize list to store generated tokens
        generated_tokens = []

        # Generate 120 tokens
        for i in range(120):
            # Create single token inputs
            position_ids = torch.tensor(
                [current_pos], dtype=torch.long, device=TEST_DEVICE
            )

            # Create single token causal mask
            single_causal_mask = causal_mask[
                :, :, current_pos : current_pos + 1, :CONTEXT_LENGTH
            ]

            # Create update mask for single token
            update_mask = torch.zeros(
                (1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            update_mask[0, 0, current_pos, 0] = 1.0

            if ENABLE_DEBUG3 and (i == 0 or i % 20 == 0):
                print(f"\nGeneration step {i}:")
                print(f"  Current position: {current_pos.item()}")
                print(f"  Position IDs: {position_ids}")
                print(f"  Single causal mask shape: {single_causal_mask.shape}")
                print(f"  Update mask shape: {update_mask.shape}")

            # Forward pass
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids[:, current_pos : current_pos + 1],
                    update_mask=update_mask,
                    position_ids=position_ids,
                    current_pos=current_pos,
                    causal_mask=single_causal_mask,
                    IN_PREFILL=False,
                )

            # Get next token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            # Break if we hit the EOT token
            if next_token_id == eot_token_id:
                print(f"\n[DEBUG] Breaking on EOT token (ID: {next_token_id})")
                break

            # Add token to generated sequence
            generated_tokens.append(next_token_id)
            total_new_tokens += 1

            # Update input_ids with the new token
            input_ids[0, current_pos + 1] = next_token_id
            current_pos += 1

        inference_end = time.time()
        inference_time = inference_end - inference_start
        inference_tokens_per_second = total_new_tokens / inference_time

        print("\n[TIMING] Inference phase:")
        print(f"  Tokens generated: {total_new_tokens}")
        print(f"  Time taken: {inference_time:.2f} seconds")
        print(f"  Speed: {inference_tokens_per_second:.2f} tokens/second")

        # Decode only the prompt and generated tokens
        prompt_text = tokenizer.decode(input_ids[0][:prompt_length])
        print(f"\nPrompt text:\n{prompt_text}")

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\nGenerated text:\n{generated_text}")

    def test_generation_with_single_prefill(self):
        """Test generation pipeline with single-token prefill approach.

        This test verifies:
        1. Tokenizer initialization and EOT token handling
        2. Prompt processing
        3. Single-token prefill approach
        4. Token-by-token generation
        5. EOT token detection and generation stopping
        6. Proper text decoding and output formatting
        """
        # Initialize tokenizer
        tokenizer = initialize_tokenizer(MODEL_PATH)
        print("\n[DEBUG] Tokenizer info:")
        print(f"  EOS token: {tokenizer.eos_token}")
        print(f"  EOS token ID: {tokenizer.eos_token_id}")
        print(f"  Special tokens: {tokenizer.special_tokens_map}")

        # Find the actual EOT token ID from the template
        test_message = [{"role": "system", "content": "test"}]
        test_ids = tokenizer.apply_chat_template(test_message, return_tensors="pt")[0]
        print("\n[DEBUG] Template token analysis:")
        print("Token sequence:")
        for i, token_id in enumerate(test_ids):
            token = tokenizer.decode([token_id])
            print(f"  {i}: {token_id} -> '{token}'")
            if "<|eot_id|>" in token:
                print(f"  Found EOT token ID: {token_id}")
                eot_token_id = token_id

        # Prepare input
        prompt = "What is Apple Neural Engine?"
        messages = [{"role": "user", "content": prompt}]
        input_ids = tokenizer.apply_chat_template(
            messages, return_tensors="pt", add_generation_prompt=True
        )

        decoded_input = tokenizer.decode(input_ids[0])
        print("\n[DEBUG] Decoded content:")
        print(f"  Prompt: {prompt}")
        print(f"  Full decoded input: {decoded_input}")

        # Ensure input_ids is on correct device
        input_ids = input_ids.to(TEST_DEVICE)
        prompt_length = input_ids.shape[1]

        # Ensure model is in eval mode and on correct device
        self.model.eval()
        self.model = self.model.to(TEST_DEVICE)

        input_ids = F.pad(
            input_ids, (0, CONTEXT_LENGTH - prompt_length), value=tokenizer.pad_token_id
        )

        if ENABLE_DEBUG3:
            print("\nInput preparation:")
            print(f"  input_ids shape: {input_ids.shape}")
            print(f"  input_ids device: {input_ids.device}")
            print(f"  input_ids dtype: {input_ids.dtype}")
            print(f"  prompt_length: {prompt_length}")
            print(f"  model device: {next(self.model.parameters()).device}")

        # Start timing prefill phase
        prefill_start = time.time()

        # Single token prefill phase
        current_pos = torch.tensor([0], device=TEST_DEVICE)

        # Create full causal mask once
        causal_mask = make_causal_mask(CONTEXT_LENGTH, 0)

        # Process each prompt token individually
        for i in range(prompt_length):
            position_ids = torch.tensor(
                [current_pos], dtype=torch.long, device=TEST_DEVICE
            )

            # Create single token causal mask
            single_causal_mask = causal_mask[
                :, :, current_pos : current_pos + 1, :CONTEXT_LENGTH
            ]

            # Create update mask for single token
            update_mask = torch.zeros(
                (1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            update_mask[0, 0, current_pos, 0] = 1.0

            if ENABLE_DEBUG3:
                print(f"\nPrefill token {i}:")
                print(f"  Current position: {current_pos.item()}")
                print(f"  Position IDs: {position_ids}")
                print(f"  Single causal mask shape: {single_causal_mask.shape}")
                print(f"  Update mask shape: {update_mask.shape}")

            # Process single token
            with torch.no_grad():
                self.model(
                    input_ids=input_ids[:, current_pos : current_pos + 1],
                    update_mask=update_mask,
                    position_ids=position_ids,
                    current_pos=current_pos,
                    causal_mask=single_causal_mask,
                    IN_PREFILL=False,
                )
            current_pos += 1

        prefill_end = time.time()
        prefill_time = prefill_end - prefill_start
        prefill_tokens_per_second = prompt_length / prefill_time
        print("\n[TIMING] Prefill phase:")
        print(f"  Tokens processed: {prompt_length}")
        print(f"  Time taken: {prefill_time:.2f} seconds")
        print(f"  Speed: {prefill_tokens_per_second:.2f} tokens/second")

        # Start timing inference phase
        inference_start = time.time()
        total_new_tokens = 0

        # Generation phase - start from position after the prompt
        current_pos = torch.tensor([prompt_length], device=TEST_DEVICE)

        # Initialize list to store generated tokens
        generated_tokens = []

        for i in range(120):  # Generate 120 tokens
            position_ids = torch.tensor(
                [current_pos], dtype=torch.long, device=TEST_DEVICE
            )

            # For single token, we need a slice of the causal mask [1, 1, 1, context_length]
            single_causal_mask = causal_mask[
                :, :, current_pos : current_pos + 1, :CONTEXT_LENGTH
            ]

            # Create update mask for single token
            update_mask = torch.zeros(
                (1, 1, CONTEXT_LENGTH, 1), dtype=MODEL_DTYPE, device=TEST_DEVICE
            )
            update_mask[0, 0, current_pos, 0] = 1.0

            if ENABLE_DEBUG3 and (i == 0 or i % 20 == 0):
                print(f"\nGeneration step {i}:")
                print(f"  Current position: {current_pos.item()}")
                print(f"  Position IDs: {position_ids}")
                print(f"  Single causal mask shape: {single_causal_mask.shape}")
                print(f"  Update mask shape: {update_mask.shape}")

            # Forward pass
            with torch.no_grad():
                logits = self.model(
                    input_ids=input_ids[:, current_pos : current_pos + 1],
                    update_mask=update_mask,
                    position_ids=position_ids,
                    current_pos=current_pos,
                    causal_mask=single_causal_mask,
                    IN_PREFILL=False,
                )

            # Get next token
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).item()

            # Break if we hit the EOT token
            if next_token_id == eot_token_id:
                print(f"\n[DEBUG] Breaking on EOT token (ID: {next_token_id})")
                break

            # Add token to generated sequence
            generated_tokens.append(next_token_id)
            total_new_tokens += 1

            # Update input_ids with the new token
            input_ids[0, current_pos + 1] = next_token_id
            current_pos += 1

        inference_end = time.time()
        inference_time = inference_end - inference_start
        inference_tokens_per_second = total_new_tokens / inference_time

        print("\n[TIMING] Inference phase:")
        print(f"  Tokens generated: {total_new_tokens}")
        print(f"  Time taken: {inference_time:.2f} seconds")
        print(f"  Speed: {inference_tokens_per_second:.2f} tokens/second")

        # Decode only the prompt and generated tokens
        prompt_text = tokenizer.decode(input_ids[0][:prompt_length])
        print(f"\nPrompt text:\n{prompt_text}")

        # Decode generated text
        generated_text = tokenizer.decode(generated_tokens)
        print(f"\nGenerated text:\n{generated_text}")


if __name__ == "__main__":
    unittest.main()
