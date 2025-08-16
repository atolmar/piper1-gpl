# scripts/06_train_model.py
# Fixed to use proper Piper training arguments

import sys
from pathlib import Path
import logging
import subprocess
from tqdm import tqdm
import re
import pandas as pd
from typing import Dict, Any, Optional

# --- Import from project utils ---
import src.utils as utils

# --- Configuration and Logger Setup ---
try:
    config = utils.load_config()
    logger = utils.setup_logger(__name__, config)
except Exception as e:
    logging.basicConfig(level=logging.CRITICAL, format='%(asctime)s - %(levelname)s - %(message)s')
    logging.critical(f"Fatal error initializing configuration or logger: {e}")
    sys.exit(1)

PIPER_SOURCE_DIR = Path.cwd()

# --- Metrics Tracking Class ---
class TrainingMetricsTracker:
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.metrics_file = output_dir / "training_metrics.csv"
        self.plot_file = output_dir / "training_summary.png"
        self.metrics: list[dict] = []
        self.patterns = {
            'epoch': re.compile(r"Epoch (\d+)"),
            'train_loss': re.compile(r"loss=([0-9.]+)"),
            'val_loss': re.compile(r"val_loss[=:]\s*([0-9.]+)")
        }
        self.current_epoch = -1

    def parse_line(self, line: str):
        epoch_match = self.patterns['epoch'].search(line)
        if epoch_match: self.current_epoch = int(epoch_match.group(1))
        train_loss_match = self.patterns['train_loss'].search(line)
        if train_loss_match and "val_loss" not in line:
            self.metrics.append({'epoch': self.current_epoch, 'type': 'train', 'loss': float(train_loss_match.group(1))})
        val_loss_match = self.patterns['val_loss'].search(line)
        if val_loss_match:
            val_loss = float(val_loss_match.group(1))
            self.metrics.append({'epoch': self.current_epoch, 'type': 'validation', 'loss': val_loss})
            logger.info(f"📊 METRICS | Epoch: {self.current_epoch}, Validation Loss: {val_loss:.5f}")

    def save_and_plot(self):
        if not self.metrics:
            logger.warning("⚠️ No metrics captured to generate summary.")
            return

        df = pd.DataFrame(self.metrics)
        df.to_csv(self.metrics_file, index=False)
        logger.info(f"✅ Training metrics saved to {self.metrics_file}")

        try:
            import matplotlib.pyplot as plt
            import numpy as np
            plt.style.use('seaborn-v0_8-whitegrid')
            fig, ax = plt.subplots(figsize=(12, 7))

            train_data = df[df['type'] == 'train'].groupby('epoch')['loss'].mean()
            val_data = df[df['type'] == 'validation'].groupby('epoch')['loss'].mean()

            train_epochs = np.array(train_data.index.tolist())
            train_losses = np.array(train_data.values.tolist())
            val_epochs = np.array(val_data.index.tolist())
            val_losses = np.array(val_data.values.tolist())

            ax.plot(train_epochs, train_losses, label='Training Loss', alpha=0.8)
            ax.plot(val_epochs, val_losses, label='Validation Loss', marker='o', linestyle='--')

            ax.set_title('Training and Validation Loss Over Epochs')
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.legend()
            fig.savefig(self.plot_file)
            logger.info(f"📈 Training summary plot saved to {self.plot_file}")
            plt.close(fig)
        except Exception as e:
            logger.error(f"❌ Failed to generate plot: {e}")

def find_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Finds the most recently modified .ckpt file in the directory."""
    checkpoints = list(checkpoint_dir.glob("*.ckpt"))
    if not checkpoints:
        return None
    return max(checkpoints, key=lambda p: p.stat().st_mtime)

def prepare_piper_dataset(dataset_dir: Path) -> Path:
    """Prepare the Piper dataset and return metadata CSV path."""
    
    # The dataset should already exist from step 5, just verify and return the metadata path
    metadata_csv = dataset_dir / "metadata.csv"
    
    if not metadata_csv.exists():
        logger.error(f"❌ Piper metadata CSV not found at {metadata_csv}")
        logger.error("Please run step 5 (validate_dataset) first to create the proper Piper dataset structure.")
        sys.exit(1)
    
    # Verify wavs directory exists (based on your metadata.csv structure)
    wavs_dir = dataset_dir / "wavs"
    
    if not wavs_dir.exists():
        logger.error(f"❌ Audio directory not found at {wavs_dir}")
        logger.error("Please run step 5 (validate_dataset) first to create the proper Piper dataset structure.")
        sys.exit(1)
    
    # Count files for verification
    wav_files = list(wavs_dir.glob("*.wav"))
    logger.info(f"✅ Found Piper dataset with {len(wav_files)} audio files in wavs/")
    
    return metadata_csv

def main():
    logger.info("🚀 Starting Stage 6: Model Training with Metrics...")
    try:
        piper_config = config.get('piper_training', {})
        
        # --- Path Management ---
        data_base_dir = Path(config.get('data_base_dir', 'data'))
        models_dir = Path(config.get('models_dir', 'models'))
        
        piper_dataset_subdir = config.get('piper_dataset_subdir', '05_piper_dataset')
        dataset_dir = data_base_dir / piper_dataset_subdir
        
        checkpoint_dir = Path(config.get('training_checkpoint_dir', 'models/checkpoints'))
        base_model_relative_path = piper_config.get('base_checkpoint_path', 'base_models/epoch=5669-step=1607900.ckpt')
        base_model_path = models_dir / base_model_relative_path
        
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        base_model_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create cache directory (Piper expects this)
        cache_dir = dataset_dir / "cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Prepare the Piper dataset (should already exist from step 5)
        piper_csv = prepare_piper_dataset(dataset_dir)
        
        # IMPORTANT: Point to the dataset directory, not the wavs subdirectory
        # Since your metadata.csv contains paths like "wavs/filename.wav",
        # Piper will automatically look in the wavs/ subdirectory
        audio_dir = dataset_dir  # This should be the parent directory of wavs/
        
        # Find checkpoint to resume from
        resume_path = find_latest_checkpoint(checkpoint_dir)
        if not resume_path:
            logger.info("ℹ️ No existing training checkpoint found. Starting from base model.")
            if not base_model_path.exists():
                logger.error(f"❌ Base model not found at {base_model_path}. Please download it first.")
                sys.exit(1)
            resume_path = base_model_path
        else:
            logger.info(f"✅ Found existing checkpoint. Resuming from: {resume_path.name}")

        # Build the proper Piper training command using absolute paths
        project_root = Path.cwd()  # Get absolute path to project root
        
        command = [
            sys.executable, "-m", "piper.train", "fit",
            "--data.csv_path", str(project_root / piper_csv),
            "--data.audio_dir", str(project_root / audio_dir),  # Point to dataset dir, not wavs/
            "--data.cache_dir", str(project_root / cache_dir),
            "--data.config_path", str(project_root / dataset_dir / "config.json"),
            "--data.voice_name", piper_config.get('voice_name', 'hungarian_voice'),
            "--data.espeak_voice", config.get('espeak_voice', 'hu'),
            "--data.batch_size", str(piper_config.get('batch_size', 8)),
            "--data.validation_split", str(piper_config.get('validation_split', 0.05)),
            "--model.sample_rate", str(piper_config.get('sample_rate', 22050)),
            "--model.learning_rate", str(piper_config.get('learning_rate', 0.00004)),
            "--trainer.max_epochs", str(piper_config.get('max_epochs', 6000)),
            "--trainer.accelerator", "auto",
            "--trainer.devices", "1",
            "--trainer.precision", "16-mixed",
            "--trainer.default_root_dir", str(project_root / checkpoint_dir),
            "--seed_everything", str(piper_config.get('seed', 42)),  # Add this line
            "--ckpt_path", str(project_root / resume_path)
        ]

        logger.info(f"🔥 Launching training process...")
        logger.info(f"   - Script CWD set to: {PIPER_SOURCE_DIR}")
        logger.info(f"   - Dataset CSV: {piper_csv}")
        logger.info(f"   - Audio directory: {audio_dir} (parent of wavs/)")
        logger.info(f"   - Cache directory: {cache_dir}")
        logger.info(f"   - Resume from: {resume_path}")

        tracker = TrainingMetricsTracker(checkpoint_dir)
        
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            bufsize=1,
            cwd=str(PIPER_SOURCE_DIR)
        )
        
        if process.stdout:
            for line in iter(process.stdout.readline, ''):
                if line:
                    print(line, end='')
                    tracker.parse_line(line)

        process.wait()

        if process.returncode == 0:
            logger.info("\n🎉 Training process completed successfully.")
            tracker.save_and_plot()
        else:
            logger.error(f"❌ Training process failed with exit code {process.returncode}.")
            sys.exit(1)

    except Exception as e:
        logger.error(f"❌ A critical error occurred: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    main()