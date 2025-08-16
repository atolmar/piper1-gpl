# scripts/08_inference_validation.py
# Refactored to use centralized utilities, configuration, load test sentences from config,
# and allow user selection of the model.

import sys
from pathlib import Path
import logging  # Import logging, but setup will be done via utils
import subprocess
from typing import Dict, Any, List
import re # For parsing model filenames if needed

# --- Import from project utils ---
# Ensure the 'src' directory is in the Python path (handled by runner or Docker)
import src.utils as utils  # Centralized utilities for config, logging

# --- Configuration and Logger Setup ---
try:
    # Load configuration using the centralized utility
    config = utils.load_config()

    # Setup logging using the centralized utility
    logger = utils.setup_logger(__name__, config)

except Exception as e:
    # If utils.setup_logger fails before creating a logger, fall back to basic config for this critical error
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"Fatal error initializing configuration or logger: {e}")
    sys.exit(1)

def find_available_models(onnx_export_dir: Path) -> List[Path]:
    """Finds available ONNX models (.onnx files) in the export directory."""
    if not onnx_export_dir.exists():
        logger.warning(f"ONNX export directory does not exist: {onnx_export_dir}")
        return []
    # Find .onnx files
    model_files = list(onnx_export_dir.glob("*.onnx"))
    # Sort by modification time, newest first (optional)
    model_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return model_files

def select_model_interactive(model_files: List[Path]) -> Path:
    """Prompts the user to select a model from a list."""
    if not model_files:
        logger.error("No ONNX models found for selection.")
        sys.exit(1)
    if len(model_files) == 1:
        logger.info(f"Only one model found, selecting automatically: {model_files[0].name}")
        return model_files[0]

    logger.info("Available ONNX Models:")
    for i, model_path in enumerate(model_files):
        # Extract a user-friendly name (filename without .onnx)
        model_name = model_path.stem
        mod_time = model_path.stat().st_mtime
        logger.info(f"  {i+1}. {model_name} (Modified: {mod_time})") # Consider formatting time

    while True:
        try:
            choice = input(f"Select model (1-{len(model_files)}): ")
            index = int(choice) - 1
            if 0 <= index < len(model_files):
                selected_model = model_files[index]
                logger.info(f"Selected model: {selected_model.name}")
                return selected_model
            else:
                logger.warning(f"Invalid choice. Please enter a number between 1 and {len(model_files)}.")
        except ValueError:
            logger.warning("Invalid input. Please enter a number.")

def main():
    """Main function to test an exported ONNX model with multiple sentences."""
    logger.info("🚀 Starting Stage 8: Inference Validation (with Model Selection)...")

    try:
        # Load configurations
        piper_config = config.get('piper_training', {})
        inference_config = config.get('inference_validation', {})

        # --- Centralized Path Management ---
        data_base_dir = Path(config.get('data_base_dir', 'data'))
        onnx_export_dir = Path(config.get('onnx_export_dir', 'models/exported_onnx'))
        inference_output_subdir = config.get('inference_output_subdir', 'models/inference_output')
        output_dir = inference_output_subdir
        #output_dir.mkdir(exist_ok=True)

        # --- Find and Select Model ---
        logger.info("Searching for available ONNX models...")
        available_models = find_available_models(onnx_export_dir)

        if not available_models:
            logger.error(f"❌ No ONNX models found in {onnx_export_dir}. Please run stage 8 (export) first.")
            sys.exit(1)

        # Select model (interactive or first one found)
        selected_model_path = select_model_interactive(available_models)
        # Derive the corresponding config path
        # Assuming Piper convention: model_name.onnx <-> model_name.onnx.json
        selected_config_path = selected_model_path.with_suffix('.onnx.json')

        # --- Validate Selected Model and Config Files Exist ---
        if not selected_model_path.exists():
            logger.error(f"❌ Selected model file does not exist: {selected_model_path}")
            sys.exit(1)
        if not selected_config_path.exists():
            logger.error(f"❌ Corresponding config file not found: {selected_config_path}")
            logger.error("The exported model may not work without its config file.")
            sys.exit(1)

        logger.info(f"✅ Using model: {selected_model_path}")
        logger.info(f"✅ Using config: {selected_config_path}")

        # --- Get Test Sentences from Config ---
        logger.info("Loading test sentences from config...")
        test_sentences: List[str] = inference_config.get('test_sentences', [])
        if not test_sentences:
            logger.warning("⚠️ No test sentences found in config['inference_validation']['test_sentences']. Using defaults.")
            test_sentences = [
                "Lenn délen, meseszép éjen édes édent remélsz.",
                "Az eső Hispániában főleg a síkon esik."
                "A szavak mély tengerén úszkálj könnyedén, mint egy darab jég."
                "Nem a tehened tehetetlen, hanem te vagy tehetetlen a teheneddel."
            ]

        logger.info("Synthesizing comprehensive Hungarian test sentences...")
        success_count = 0
        # Use the stem of the selected model for output file naming
        model_name_stem = selected_model_path.stem

        for i, sentence in enumerate(test_sentences, 1):
            # Name output files based on selected model
            output_wav_path = Path(output_dir) / f"{model_name_stem}_sample_{i}.wav"
            logger.info(f"[{i}/{len(test_sentences)}] Synthesizing: '{sentence}'")

            # --- Construct the Piper TTS command ---
            command = [
                sys.executable, "-m", "piper",
                "--model", str(selected_model_path),
                "--config", str(selected_config_path),
                "--output_file", str(output_wav_path),
                "--length-scale", "1.3",  # Slows down speech (e.g., 1.2 = 120% of original duration)
                "--noise-scale", "0.7", # Adjusts variability (default often around 0.667)
                "--noise-w", "0.8"        # Adjusts prosody (default often around 0.8)
            ]

            # Run Piper TTS
            process = subprocess.Popen(command, stdin=subprocess.PIPE, text=True, encoding='utf-8')
            _, stderr = process.communicate(input=sentence) # Capture stderr

            if process.returncode == 0 and output_wav_path.exists():
                logger.info(f"✅ Generated: {output_wav_path}")
                success_count += 1
            else:
                logger.error(f"❌ Failed to generate audio for sentence {i}")
                if stderr:
                    logger.error(f"Stderr: {stderr}")

        if success_count > 0:
            logger.info(f"\n🎉 Generated {success_count}/{len(test_sentences)} test samples!")
            logger.info(f"📁 Audio files are in: {output_dir}")
            logger.info("🎧 Listen to evaluate your Piper voice model quality.")
        else:
            logger.error("❌ All generations failed. Check Piper installation and model files.")
            sys.exit(1)

    except KeyboardInterrupt:
        logger.info("\n🛑 Inference validation interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()