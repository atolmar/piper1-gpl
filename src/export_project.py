# export_project.py
import os
import glob
from datetime import datetime
from pathlib import Path
import fnmatch
import sys
from typing import List, Optional

# Import project utilities
try:
    from src.utils import load_config, setup_logger, DirectoryManager
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    print("Warning: Project utils not available. Running in standalone mode.")

def export_project(base_dir=".", logs_dir=None, config_path='config/config.yaml'):
    """
    Scans the project directory and concatenates the content of specified files
    into a single text file for easy review or sharing with AI coding assistants.
    
    Args:
        base_dir (str): The base directory to scan (default: current directory)
        logs_dir (str): Directory to save the export file (default: from config or "logs")
        config_path (str): Path to configuration file (default: "config/config.yaml")
    """
    
    # Initialize configuration and logging if utils are available
    config = {}
    logger = None
    dir_manager = None
    
    if UTILS_AVAILABLE:
        try:
            config = load_config(config_path)
            logger = setup_logger('export_project', config)
            dir_manager = DirectoryManager(config)
            dir_manager.ensure_all_dirs_exist()
            logger.info("Project export started with configuration loaded")
        except Exception as e:
            print(f"Warning: Could not load project config: {e}")
            print("Continuing in standalone mode...")
    
    # Determine logs directory
    if logs_dir is None:
        if dir_manager:
            logs_path = dir_manager.logs_dir
        else:
            logs_path = Path("logs")
    else:
        logs_path = Path(logs_dir)
    
    logs_path.mkdir(exist_ok=True)
    
    # Generate timestamped filename
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    project_name = Path(base_dir).resolve().name
    output_filename = logs_path / f"{project_name}_export_{timestamp}.txt"
    
    if logger:
        logger.info(f"Export file will be saved as: {output_filename}")
    
    # --- Configuration: Define all files and directories to be exported ---
    
    # Static files to include (if they exist)
    static_files = [
        '.gitignore',
        'requirements.txt',
        'README.md',
        'README.rst',
        'pyproject.toml',
        'setup.py',
        'setup.cfg',
        'run.py',
        'main.py',
        'app.py',
        'Dockerfile',
        'docker-compose.yml',
        'docker-compose.yaml',
        '.env.example',
        'config.ini',
        'config.yaml',
        'config.json',
        'package.json',
        'Makefile',
        'LICENSE',
        'CHANGELOG.md',
        'CONTRIBUTING.md',
    ]
    
    # Directories to scan for code files
    code_directories = [
        'src',
        'scripts',
        'misc',
        'config',
        'tests',
        'docs',
        'utils',
        'lib',
        'modules',
        'components',
    ]
    
    # File extensions to include when scanning directories
    code_extensions = [
        '*.py',
        '*.js',
        '*.ts',
        '*.jsx',
        '*.tsx',
        '*.java',
        '*.cpp',
        '*.c',
        '*.h',
        '*.hpp',
        '*.cs',
        '*.go',
        '*.rs',
        '*.php',
        '*.rb',
        '*.swift',
        '*.kt',
        '*.scala',
        '*.r',
        '*.R',
        '*.m',
        '*.sh',
        '*.bat',
        '*.ps1',
        '*.yaml',
        '*.yml',
        '*.json',
        '*.xml',
        '*.toml',
        '*.ini',
        '*.cfg',
        '*.conf',
        '*.sql',
        '*.md',
        '*.rst',
        '*.txt',
    ]
    
    # Files and patterns to exclude
    exclude_patterns = [
        '__pycache__',
        '*.pyc',
        '*.pyo',
        '*.pyd',
        '.git',
        '.svn',
        '.hg',
        'node_modules',
        '.pytest_cache',
        '.coverage',
        'coverage.xml',
        '*.log',
        '.DS_Store',
        'Thumbs.db',
        '*.tmp',
        '*.temp',
        '.env',
        '.venv',
        'venv',
        'env',
        'build',
        'dist',
        '*.egg-info',
        '.tox',
        '.cache',
        '.mypy_cache',
    ]
    
    def should_exclude(file_path):
        """Check if a file should be excluded based on exclude patterns."""
        file_str = str(file_path)
        for pattern in exclude_patterns:
            if fnmatch.fnmatch(file_str, pattern) or pattern in file_str:
                return True
        return False
    
    def get_file_size(file_path):
        """Get human-readable file size."""
        try:
            size = os.path.getsize(file_path)
            for unit in ['B', 'KB', 'MB', 'GB']:
                if size < 1024.0:
                    return f"{size:.1f} {unit}"
                size /= 1024.0
            return f"{size:.1f} TB"
        except:
            return "Unknown"
    
    def log_message(level, message):
        """Log message using logger if available, otherwise print."""
        if logger:
            getattr(logger, level)(message)
        else:
            print(f"{level.upper()}: {message}")
    
    log_message('info', f"Starting project export from: {Path(base_dir).resolve()}")
    log_message('info', f"Scanning for files in directories: {', '.join(code_directories)}")
    
    # --- Collect all files to export ---
    files_to_export = []
    
    log_message('info', "Collecting static configuration files...")
    # Add static files that exist
    for file_name in static_files:
        file_path = Path(base_dir) / file_name
        if file_path.exists() and file_path.is_file() and not should_exclude(file_path):
            files_to_export.append(file_path)
            log_message('debug', f"Added static file: {file_path}")
    
    log_message('info', f"Found {len(files_to_export)} static files")
    
    # Scan code directories
    log_message('info', "Scanning code directories...")
    for directory in code_directories:
        dir_path = Path(base_dir) / directory
        if dir_path.exists() and dir_path.is_dir():
            log_message('debug', f"Scanning directory: {dir_path}")
            files_found_in_dir = 0
            for extension in code_extensions:
                pattern = str(dir_path / "**" / extension)
                for file_path in glob.glob(pattern, recursive=True):
                    file_path = Path(file_path)
                    if file_path.is_file() and not should_exclude(file_path):
                        files_to_export.append(file_path)
                        files_found_in_dir += 1
            log_message('debug', f"Found {files_found_in_dir} files in {directory}")
    
    # Remove duplicates and sort
    files_to_export = sorted(list(set(files_to_export)))
    log_message('info', f"Total files to export: {len(files_to_export)}")
    
    # --- Export files ---
    log_message('info', f"Starting export to: {output_filename}")
    try:
        with open(output_filename, 'w', encoding='utf-8') as f_out:
            # --- Write Header ---
            header = f"""# {project_name} - Project Export
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Base Directory: {Path(base_dir).resolve()}
# Total Files: {len(files_to_export)}
# Configuration: {'Loaded' if config else 'Standalone mode'}
# 
# This file contains the complete source code and configuration files
# for AI coding assistants to understand the project structure and codebase.
#
# ============================================================================

"""
            f_out.write(header)
            
            # --- Write Table of Contents ---
            log_message('info', "Writing table of contents...")
            f_out.write("# TABLE OF CONTENTS\n")
            f_out.write("# ==================\n\n")
            for i, file_path in enumerate(files_to_export, 1):
                relative_path = file_path.relative_to(base_dir)
                file_size = get_file_size(file_path)
                f_out.write(f"# {i:3d}. {relative_path} ({file_size})\n")
            f_out.write("\n" + "="*80 + "\n\n")
            
            # --- Process and Append Each File ---
            log_message('info', "Processing and writing files...")
            for i, file_path in enumerate(files_to_export, 1):
                relative_path = file_path.relative_to(base_dir)
                normalized_path = str(relative_path).replace(os.sep, '/')
                file_size = get_file_size(file_path)
                
                log_message('debug', f"Processing file {i}/{len(files_to_export)}: {normalized_path}")
                
                # Write file header
                f_out.write(f"# FILE {i}/{len(files_to_export)}: {normalized_path}\n")
                f_out.write(f"# Size: {file_size}\n")
                f_out.write(f"# {'='*60}\n\n")
                
                try:
                    with open(file_path, 'r', encoding='utf-8', errors='ignore') as f_in:
                        content = f_in.read().strip()
                        if content:
                            f_out.write(content)
                        else:
                            f_out.write("# [Empty file]")
                except FileNotFoundError:
                    error_msg = f"File not found: {normalized_path}"
                    f_out.write(f"# !!! {error_msg} !!!")
                    log_message('warning', error_msg)
                except PermissionError:
                    error_msg = f"Permission denied: {normalized_path}"
                    f_out.write(f"# !!! {error_msg} !!!")
                    log_message('warning', error_msg)
                except UnicodeDecodeError:
                    error_msg = f"Binary file (not readable as text): {normalized_path}"
                    f_out.write(f"# !!! {error_msg} !!!")
                    log_message('debug', error_msg)
                except Exception as e:
                    error_msg = f"Error reading {normalized_path}: {e}"
                    f_out.write(f"# !!! {error_msg} !!!")
                    log_message('error', error_msg)
                
                f_out.write("\n\n" + "="*80 + "\n\n")
        
        # --- Success message with statistics ---
        total_size = sum(os.path.getsize(f) for f in files_to_export if os.path.exists(f))
        size_mb = total_size / (1024 * 1024)
        
        success_msg = f"Project exported successfully! Files: {len(files_to_export)}, Size: {size_mb:.2f} MB"
        log_message('info', success_msg)
        
        print(f"✅ {success_msg}")
        print(f"📁 Output file: {output_filename}")
        print(f"🕐 Export completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
        return str(output_filename)
        
    except IOError as e:
        error_msg = f"Could not write to file {output_filename}: {e}"
        log_message('error', error_msg)
        print(f"❌ Error: {error_msg}")
        return None
    except Exception as e:
        error_msg = f"Unexpected error during export: {e}"
        log_message('error', error_msg)
        print(f"❌ {error_msg}")
        return None

def main():
    """Main function to run the export."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Export project files to a single text file for AI coding assistants"
    )
    parser.add_argument(
        '--base-dir', '-d',
        default='.',
        help='Base directory to scan (default: current directory)'
    )
    parser.add_argument(
        '--logs-dir', '-l',
        default=None,
        help='Directory to save export file (default: from config or "logs")'
    )
    parser.add_argument(
        '--config', '-c',
        default='config/config.yaml',
        help='Path to configuration file (default: config/config.yaml)'
    )
    
    args = parser.parse_args()
    
    export_project(args.base_dir, args.logs_dir, args.config)

if __name__ == "__main__":
    main()