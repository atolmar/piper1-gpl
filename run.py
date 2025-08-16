# run.py
import sys
import subprocess
import argparse
from typing import List, Set
import os
from pathlib import Path

from src.utils import load_config, setup_logger

def parse_steps(steps_str: str) -> Set[str]:
    """Parses a comma-separated string of steps/ranges (e.g., '1,3-5') into a set of step numbers."""
    steps = set()
    parts = steps_str.split(',')
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                steps.update(map(str, range(start, end + 1)))
            except ValueError:
                raise ValueError(f"Invalid range in --steps argument: {part}")
        else:
            try:
                steps.add(str(int(part)))
            except ValueError:
                raise ValueError(f"Invalid step number in --steps argument: {part}")
    return steps

def main():
    """Main pipeline orchestrator."""
    config = load_config()
    logger = setup_logger('pipeline_orchestrator', config)

    # --- Pipeline Stage Definitions ---
    # Maps step number to the script path
    # Corrected to align with standard file naming and project_export.txt mapping
    PIPELINE_STAGES = {
        '0': 'scripts/00_setup_project.py',
        '1': 'scripts/01_normalize_text.py',
        '2': 'scripts/02_preprocess_audio.py',
        '3': 'scripts/03_align_audio.py',
        '4': 'scripts/04_segment_audio.py',
        '5': 'scripts/05_validate_dataset.py',
        '6': 'scripts/06_train_model.py',
        '7': 'scripts/07_export_model.py',    
        '8': 'scripts/08_inference_validation.py'           
    }


    parser = argparse.ArgumentParser(description="Apoka TTS Pipeline Orchestrator")
    parser.add_argument(
        '--steps',
        type=str,
        help="Comma-separated list of steps or ranges to run (e.g., '1,2,5' or '3-6'). Runs all steps if not specified."
    )
    args = parser.parse_args()

    if args.steps:
        try:
            steps_to_run = parse_steps(args.steps)
            ordered_steps_to_run = sorted(steps_to_run, key=int)
        except ValueError as e:
            logger.error(f"❌ Invalid --steps argument: {e}")
            sys.exit(1)
    else:
        ordered_steps_to_run = sorted(PIPELINE_STAGES.keys(), key=int)
    
    logger.info(f"🚀 Starting Apoka TTS Pipeline. Steps to run: {', '.join(ordered_steps_to_run)}")

    # --- Prepare Environment and Working Directory for Subprocesses ---
    # Ensure PYTHONPATH includes the project root so 'src' is findable
    env = os.environ.copy()
    project_root_path = Path(__file__).resolve().parent
    project_root_str = str(project_root_path)
    current_pythonpath = env.get('PYTHONPATH', '')
    if current_pythonpath:
        env['PYTHONPATH'] = project_root_str + os.pathsep + current_pythonpath
    else:
        env['PYTHONPATH'] = project_root_str
    logger.debug(f"Set PYTHONPATH for subprocesses to: {env['PYTHONPATH']}")
    
    # The cwd for subprocesses MUST be the project root.
    # This is crucial for resolving relative paths (like 'import utils' finding 'src/utils.py')
    # and for scripts that expect to be run from the project root.
    subprocess_cwd = project_root_str
    logger.debug(f"Set working directory (cwd) for subprocesses to: {subprocess_cwd}")
    # --- End Environment & CWD Setup ---

    for step in ordered_steps_to_run:
        if step not in PIPELINE_STAGES:
            logger.warning(f"⚠️ Step {step} is not defined. Skipping.")
            continue
        
        script_path = PIPELINE_STAGES[step]
        logger.info(f"\n{'='*20} RUNNING STAGE {step}: {script_path} {'='*20}")
        
        try:
            # --- Key Fix: Pass env AND set cwd ---
            # Use sys.executable to ensure we run the script with the same Python interpreter (from our venv)
            # Pass the modified environment (with PYTHONPATH) AND
            # Crucially, set the working directory (cwd) to the project root.
            result = subprocess.run(
                [sys.executable, script_path],
                check=True,
                env=env,           # Pass the environment with PYTHONPATH
                cwd=subprocess_cwd # Set the working directory to project root
            )
            
            # Redundant check removed: if result.returncode != 0:
            # This block was unreachable because subprocess.run with check=True raises
            # CalledProcessError if returncode is non-zero.
            # The except block below handles this case.
            
            logger.info(f"✅ Stage {step} completed successfully.")

        except subprocess.CalledProcessError:
            logger.error(f"❌ Stage {step} ({script_path}) failed. Halting pipeline.")
            sys.exit(1)
        except FileNotFoundError:
            logger.error(f"❌ Script not found: {script_path}. Halting pipeline.")
            sys.exit(1)

    logger.info("\n🎉 Pipeline finished successfully!")


if __name__ == "__main__":
    main()