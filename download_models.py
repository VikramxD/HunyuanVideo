import subprocess
import os

def run_command(command):
    """
    Execute a shell command and raise an exception if it fails.

    Args:
        command (str): The shell command to execute.

    Raises:
        RuntimeError: If the command exits with a non-zero status.
    """
    result = subprocess.run(command, shell=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed: {command}")

def install_huggingface_cli():
    """
    Install the Huggingface CLI using pip if it's not already installed.
    """
    print("Installing Huggingface CLI...")
    run_command("uv pip install 'huggingface_hub[cli]'")

def download_model(repo_name, local_dir):
    """
    Download a model from the Huggingface repository using huggingface-cli.

    Args:
        repo_name (str): The Huggingface repository name (e.g., 'tencent/HunyuanVideo').
        local_dir (str): The local directory to store the downloaded model.
    """
    print(f"Downloading {repo_name} to {local_dir}...")
    command = f"huggingface-cli download {repo_name} --local-dir {local_dir}"
    run_command(command)

def preprocess_text_encoder(input_dir, output_dir):
    """
    Preprocess the text encoder model for optimized GPU memory usage.

    Args:
        input_dir (str): Directory containing the original model.
        output_dir (str): Directory to save the preprocessed model.
    """
    print(f"Preprocessing text encoder from {input_dir} to {output_dir}...")
    script_path = "hyvideo/utils/preprocess_text_encoder_tokenizer_utils.py"
    command = f"python {script_path} --input_dir {input_dir} --output_dir {output_dir}"
    run_command(command)

def main():
    """
    Main script to download the HunyuanVideo model, MLLM text encoder, and CLIP text encoder.
    """
    base_dir = "ckpts"
    os.makedirs(base_dir, exist_ok=True)

    try:
        # Step 1: Install Huggingface CLI
        install_huggingface_cli()

        # Step 2: Download HunyuanVideo model
        download_model("tencent/HunyuanVideo", base_dir)

        # Step 3: Download and preprocess MLLM model
        mllm_repo = "xtuner/llava-llama-3-8b-v1_1-transformers"
        mllm_dir = os.path.join(base_dir, "llava-llama-3-8b-v1_1-transformers")
        download_model(mllm_repo, mllm_dir)

        text_encoder_dir = os.path.join(base_dir, "text_encoder")
        preprocess_text_encoder(mllm_dir, text_encoder_dir)

        # Step 4: Download CLIP model
        clip_repo = "openai/clip-vit-large-patch14"
        clip_dir = os.path.join(base_dir, "text_encoder_2")
        download_model(clip_repo, clip_dir)

        print("All models downloaded and preprocessed successfully!")

    except RuntimeError as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
