# Copyright (c) Kyutai, all rights reserved.
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import os
import time
from pathlib import Path
from typing import Optional

import torch
import typer
import uvicorn # Import uvicorn
from typing_extensions import Annotated

from hibiri.client_utils import audio_read, log, save_results, stack_and_pad_audio
from hibiri.model import HibiriTranslator, seed_all # Import HibiriTranslator and seed_all
from hibiri.server import app as fastapi_app # Import the FastAPI app

MODULE_DIR: Path = Path(__file__).parent
DEFAULT_REPO: str = "kyutai/hibiki-zero-3b-pytorch-bf16@23b3e0b41782026c81dd5283a034107b01f9e513"
DEFAULT_AUDIO_SAMPLES: list[Path] = [
    MODULE_DIR / "samples" / fname for fname in os.listdir(MODULE_DIR / "samples")
]
DEFAULT_STATIC_DIR = MODULE_DIR / "static" # This might be removed if frontend is separate

cli_app = typer.Typer()


@cli_app.command()
@torch.no_grad()
def serve(
    host: Annotated[str, typer.Option(help="Host to bind the server to.")] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Port to bind the server to.")] = 8000,
    ssl: Annotated[
        Optional[str], typer.Option(help="Directory containing cert.pem and key.pem.")
    ] = None,
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
):
    # sanity checks
    if not torch.cuda.is_available():
        log(
            "error",
            "Found no NVIDIA driver on your system. The server needs to be launched from a machine that has access to a GPU.",
        )
        return

    seed_all(seed) # Use seed_all from hibiri.model

    log("info", "Starting Hibiri server (FastAPI).")

    # SSL context for uvicorn
    ssl_keyfile = None
    ssl_certfile = None
    if ssl is not None:
        ssl_keyfile = os.path.join(ssl, "key.pem")
        ssl_certfile = os.path.join(ssl, "cert.pem")
        if not Path(ssl_keyfile).exists() or not Path(ssl_certfile).exists():
            log("error", f"SSL certificate or key not found in {ssl}")
            return
    
    # Run the FastAPI app using uvicorn
    uvicorn.run(
        fastapi_app,
        host=host,
        port=port,
        ssl_keyfile=ssl_keyfile,
        ssl_certfile=ssl_certfile,
        log_level="info",
    )


@cli_app.command(name="translate")
@torch.no_grad()
def generate_or_translate(
    files: Annotated[list[Path], typer.Option("--file", help="Input files to translate.")] = None,
    gen_duration: Annotated[
        float,
        typer.Option(
            help="Generation duration in seconds. Should be <=120 seconds for Hibiri-Zero."
        ),
    ] = 120,
    out_dir: Annotated[Path, typer.Option(help="Directory where to save the outputs.")] = Path(
        "./translations"
    ),
    tag: Annotated[
        str, typer.Option(help="Tag to add to translation outputs filenames to identify them.")
    ] = None,
    repeats: Annotated[int, typer.Option(help="Do repeats generation for each input file.")] = 1,
    hf_repo: Annotated[
        str, typer.Option(help="HF repo for model, codec and text tokenizer.")
    ] = DEFAULT_REPO,
    config_path: Annotated[Optional[str], typer.Option(help="Path to a config file.")] = None,
    tokenizer: Annotated[Optional[str], typer.Option(help="Path to a text tokenizer file.")] = None,
    model_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Hibiki-Zero checkpoint.")
    ] = None,
    mimi_weight: Annotated[
        Optional[str], typer.Option(help="Path to a Mimi codec checkpoint.")
    ] = None,
    lora_weight: Annotated[Optional[str], typer.Option(help="Path to a LoRA checkpoint.")] = None,
    fuse_lora: Annotated[
        bool, typer.Option("--fuse-lora/--no-fuse-lora", help="Fuse LoRA layers.")
    ] = True,
    bf16: Annotated[bool, typer.Option(help="Use bfloat16.")] = False,
    device: Annotated[str, typer.Option(help="Device to run on.")] = "cuda",
    seed: Annotated[int, typer.Option(help="Random seed.")] = 42,
    source_lang: Annotated[str, typer.Option(help="Source language for translation.")] = "es",
    target_lang: Annotated[str, typer.Option(help="Target language for translation.")] = "en",
):
    if not torch.cuda.is_available():
        log(
            "error",
            "Found no NVIDIA driver on your system. Generation needs to be done on a machine that has access to a GPU.",
        )
        return

    seed_all(seed)

    log("info", "Starting Hibiri batch translation.")
    files = files if files is not None else DEFAULT_AUDIO_SAMPLES
    files = [fpath for fpath in files for _ in range(repeats)]
    all_files_exist: bool = len(files) > 0
    for fpath in files:
        if not fpath.exists():
            log("error", f"File not found: {fpath}")
            all_files_exist = False
    if not all_files_exist:
        if len(files) == 0:
            log("error", "No files provided.")
        return
    log("info", "The following audios will be processed:")
    for fidx, fpath in enumerate(files):
        log("info", f"{fidx} : " + "{0}", [(fpath, "grey")])

    translator = HibiriTranslator(
        model_path=hf_repo,
        device=device,
        bf16=bf16,
    )

    log("info", "Loading audios and performing translation...")
    translated_outputs = [] # Will store list of (audio_tensor, text_str)
    input_wavs_for_save = [] # Will store (fpath, wav_tensor) for save_results

    for fpath in files:
        target_sample_rate_for_model = translator.mimi.sample_rate
        wav_tensor, sr = audio_read(fpath, to_sample_rate=target_sample_rate_for_model, mono=True)
        
        input_wavs_for_save.append((fpath, wav_tensor))

        wav_tensor_float = wav_tensor.to(dtype=torch.float32) / 32768.0 if wav_tensor.dtype == torch.int16 else wav_tensor

        translated_audio_tensor, translated_text = translator.translate(
            wav_tensor_float,
            source_lang=source_lang,
            target_lang=target_lang,
        )
        
        translated_outputs.append((translated_audio_tensor, translated_text))

    adapted_outputs_for_save = []
    for audio_tensor, text_str in translated_outputs:
        audio_np_int16 = (audio_tensor.cpu().numpy() * 32768).astype(np.int16)
        adapted_outputs_for_save.append((audio_np_int16, text_str))

    save_results(
        inputs=zip([f[0] for f in input_wavs_for_save], [f[1] for f in input_wavs_for_save]),
        outputs=adapted_outputs_for_save,
        sample_rate=target_sample_rate_for_model,
        output_dir=out_dir,
        tag=tag,
    )
    log("info", "Saved translation results in {0}", [(out_dir, "green")])


def main():
    cli_app()


if __name__ == "__main__":
    main()