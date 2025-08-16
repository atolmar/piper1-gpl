# src/utils.py
import yaml
import logging
import sys
from pathlib import Path
from typing import Dict, Any, Optional

def load_config(config_path: str = 'config/config.yaml') -> Dict[str, Any]:
    """Loads the YAML configuration file from the project root."""
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = Path.cwd() / config_file
    try:
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
            if config is None:
                print(f"FATAL: Config file at '{config_path}' is empty.", file=sys.stderr)
                sys.exit(1)
            return config
    except FileNotFoundError:
        print(f"FATAL: Config file not found at '{config_path}'.", file=sys.stderr)
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL: Error parsing YAML config: {e}", file=sys.stderr)
        sys.exit(1)

def setup_logger(name: str, config: Dict[str, Any]) -> logging.Logger:
    """Sets up a configured logger for a specific pipeline stage."""
    logs_dir = Path(config.get('logs_dir', 'logs'))
    logs_dir.mkdir(exist_ok=True)
    log_file = logs_dir / f"{Path(name).stem}.log"
    
    logger = logging.getLogger(name)
    level = getattr(logging, config.get('log_level', 'INFO').upper(), logging.INFO)
    logger.setLevel(level)
    
    if logger.hasHandlers():
        logger.handlers.clear()
        
    fh = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(fh)
    
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
    logger.addHandler(ch)
    
    return logger

class DirectoryManager:
    """A centralized handler for all project directory paths."""
    def __init__(self, config: Dict[str, Any]):
        self.project_root = Path().cwd()
        self.data_base_dir = self.project_root / config.get('data_base_dir', 'data')
        self.models_dir = self.project_root / config.get('models_dir', 'models')
        self.raw_audio = self.data_base_dir / '01_raw' / 'audio'
        self.raw_text = self.data_base_dir / '01_raw' / 'text'
        self.normalized_audio = self.data_base_dir / '02_normalized' / 'audio'
        self.normalized_text = self.data_base_dir / '02_normalized' / 'text'
        self.aligned = self.data_base_dir / '03_aligned'
        self.segmented = self.data_base_dir / '04_segmented'
        self.segmented_wavs = self.segmented / 'wavs'
        self.piper_dataset = self.data_base_dir / '05_piper_dataset'
        self.checkpoints = self.models_dir / 'checkpoints'
        self.onnx_export = self.models_dir / 'exported_onnx'
        self.logs_dir = self.project_root / config.get('logs_dir', 'logs')

    def ensure_all_dirs_exist(self):
        """Creates all necessary directories if they don't exist."""
        for path in self.__dict__.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)
                
    # --- FIX: Integrated your corrected type hint ---
    def validate_required_dirs(self, required_dirs: Optional[list[str]] = None) -> bool:
        """Validates that required directories exist and are accessible."""
        if required_dirs is None:
            required_dirs = ['raw_audio', 'raw_text']
        
        for dir_name in required_dirs:
            if hasattr(self, dir_name):
                path = getattr(self, dir_name)
                if not path.exists() or not path.is_dir():
                    print(f"ERROR: Required directory '{path}' not found or is not a directory.", file=sys.stderr)
                    return False
        return True