# Prepare for Hugging Face Upload

The `prepare_hf.sh` script automates the preparation of converted ANEMLL models for uploading to Hugging Face.

## Prerequisites

1. **Hugging Face Account and Token**
   - Create an account at [Hugging Face](https://huggingface.co)
   - Generate an access token at https://huggingface.co/settings/tokens
   - Set up your token:
     ```bash
     # Option 1: Set environment variable
     export HUGGING_FACE_HUB_TOKEN=your_token_here
     
     # Option 2: Login via CLI
     huggingface-cli login
     ```

2. **Hugging Face CLI**
   ```bash
   pip install huggingface_hub
   ```

## Usage

```bash
./anemll/utils/prepare_hf.sh --input <converted_model_dir> [--output <output_dir>] [--org <org>] [--ios]
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--input` | Directory containing converted model files (with meta.yaml) | Yes |
| `--output` | Output directory for HF distribution (defaults to input_dir/hf_dist) | No |
| `--org` | Hugging Face organization/account (defaults to anemll) | No |
| `--ios` | prepare iOS-ready version with unzipped MLMODELC files in {output}/ios folder | No |

### Example

```bash
# Prepare standard distribution
./anemll/utils/prepare_hf.sh --input /path/to/converted/model

# Prepare both standard and iOS distributions
./anemll/utils/prepare_hf.sh --input /path/to/converted/model --ios

# Custom output and organization
./anemll/utils/prepare_hf.sh \
    --input /path/to/converted/model \
    --output /path/to/hf/dist \
    --org myorganization \
    --ios
```

## What the Script Does

1. **Reads Configuration**
   - Extracts model information from meta.yaml
   - Gets model name, context length, batch size, etc.

2. **Prepares Model Files**
   - Standard distribution: Compresses .mlmodelc directories into zip files
   - iOS distribution (optional): Copies uncompressed .mlmodelc directories
   - Handles embeddings, LM head, and FFN/prefill chunks

3. **Copies Required Files**
   - meta.yaml
   - config.json (required for offline iOS tokenizer)
   - tokenizer.json
   - tokenizer_config.json
   - chat.py and chat_full.py

> [!Important]
> The `config.json` file is required for iOS tokenizer to work offline without internet connection. 
> Make sure this file is present in your converted model directory before preparing for distribution.

4. **Generates README.md**
   - Uses readme.template
   - Replaces placeholders with actual values

## Output Structure

### Standard Distribution

```
output_directory/
├── standard/
│   ├── meta.yaml
│   ├── tokenizer.json
│   ├── tokenizer_config.json
│   ├── chat.py
│   ├── chat_full.py
│   ├── README.md
│   ├── llama_embeddings_lutX.mlmodelc.zip
│   ├── llama_lm_head_lutX.mlmodelc.zip
│   └── llama_FFN_PF_lutX_chunk_XXofYY.mlmodelc.zip
└── ${FULL_MODEL_NAME}.zip
```

### iOS Distribution

```
output_directory/ios/
├── meta.yaml
├── tokenizer.json
├── tokenizer_config.json
├── config.json           # Required for iOS offline tokenizer
├── chat.py               # Optional, for debug/testing in macOS
├── chat_full.py          # Optional, for debug/testing in macOS
├── llama_embeddings_lutX.mlmodelc/  # Uncompressed directory
├── llama_lm_head_lutX.mlmodelc/  # Uncompressed directory
└── llama_FFN_PF_lutX_chunk_NNofMM.mlmodelc/  # Uncompressed directory
```

Note that the iOS distribution contains **uncompressed** `.mlmodelc` directories, which are required for iOS apps to load the models directly. The `chat.py` and `chat_full.py` files are optional and included only for debugging and testing in macOS environments.

## Uploading to Hugging Face

After running the script, you can upload the model using:

```bash
# First, ensure you're logged in
huggingface-cli login

# For standard distribution:
huggingface-cli upload <org>/<model-name>_<version> <output_directory>/standard

# For iOS distribution:
huggingface-cli upload <org>/<model-name>_<version> <output_directory>/ios

# Example with real paths:
# Standard: huggingface-cli upload anemll/anemll-Meta-Llama-3.2-1B-ctx512_0.3.0 /path/to/hf_dist/standard
# iOS: huggingface-cli upload anemll/anemll-Meta-Llama-3.2-1B-ctx512_0.3.0 /path/to/hf_dist/ios

# To update just the README file in an existing repository:
huggingface-cli upload <org>/<model-name>_<version> <output_directory>/standard/README.md  # For standard distribution
huggingface-cli upload <org>/<model-name>_<version> <output_directory>/ios/README.md      # For iOS distribution
```

### Creating a New Model Repository

If this is your first time uploading this model:

1. Go to https://huggingface.co/new
2. Create a new model repository with name matching the FULL model name (including version)
   Example: `anemll-Meta-Llama-3.2-1B-ctx512_0.1.1`
3. Set visibility (public/private)
4. Then run the upload command

### Troubleshooting

1. **Token Issues**
   ```bash
   # Check if token is set
   echo $HUGGING_FACE_HUB_TOKEN
   
   # Re-login if needed
   huggingface-cli login
   ```

2. **Permission Issues**
   - Ensure you're a member of the 'anemll' organization
   - Check repository visibility settings
   - Verify write permissions

3. **Size Issues**
   - Hugging Face has a file size limit
   - Large models may need Git LFS
   - Contact Hugging Face support for increased limits

## See Also

- [Model Conversion Guide](convert_model.md)
- [Chat Interface Documentation](chat.md)
- [Hugging Face Documentation](https://huggingface.co/docs) 