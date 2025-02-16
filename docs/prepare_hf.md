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
./anemll/utils/prepare_hf.sh --input <converted_model_dir> [--output <output_dir>] [--org <org>]
```

### Parameters

| Parameter | Description | Required |
|-----------|-------------|----------|
| `--input` | Directory containing converted model files (with meta.yaml) | Yes |
| `--output` | Output directory for HF distribution (defaults to input_dir/hf_dist) | No |
| `--org` | Hugging Face organization/account (defaults to anemll) | No |

### Example

```bash
# Prepare model for upload with default organization (anemll)
./anemll/utils/prepare_hf.sh --input /path/to/converted/model

# Specify custom output directory
./anemll/utils/prepare_hf.sh \
    --input /path/to/converted/model \
    --output /path/to/hf/dist

# Use custom Hugging Face organization
./anemll/utils/prepare_hf.sh \
    --input /path/to/converted/model \
    --org myorganization
```

## What the Script Does

1. **Reads Configuration**
   - Extracts model information from meta.yaml
   - Gets model name, context length, batch size, etc.

2. **Compresses Model Files**
   - Compresses all .mlmodelc directories into zip files
   - Handles embeddings, LM head, and FFN/prefill chunks

3. **Copies Required Files**
   - meta.yaml
   - tokenizer.json
   - tokenizer_config.json
   - chat.py and chat_full.py

4. **Generates README.md**
   - Uses readme.template
   - Replaces placeholders with actual values

## Output Structure

```
output_directory/
├── meta.yaml
├── tokenizer.json
├── tokenizer_config.json
├── chat.py
├── chat_full.py
├── README.md
├── llama_embeddings.mlmodelc.zip
├── llama_lm_head_lutX.mlmodelc.zip
└── llama_FFN_PF_lutX_chunk_NNofMM.mlmodelc.zip
```

## Uploading to Hugging Face

After running the script, you can upload the model using:

```bash
# First, ensure you're logged in
huggingface-cli login

# Then upload using the command printed by the script
# The repository name will include both the model name and version number
# For example: anemll/anemll-Meta-Llama-3.2-1B-ctx512_0.2.0
huggingface-cli upload <org>/<model-name>_<version> output_directory

# To update just the README file in an existing repository:
huggingface-cli upload <org>/<model-name>_<version> output_directory/README.md
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