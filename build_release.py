"""
Build script to create a release package for Touhou Character Classifier.
Run this to generate a zip file ready for GitHub release.
"""

import shutil
import zipfile
from pathlib import Path

def create_release_package():
    # Paths
    root = Path(__file__).parent
    release_dir = root / "release_build"
    package_name = "TouhouCharacterClassifier"
    package_dir = release_dir / package_name
    
    # Clean previous build
    if release_dir.exists():
        shutil.rmtree(release_dir)
    
    package_dir.mkdir(parents=True)
    
    # Files to include
    files_to_copy = [
        "app.py",
        "model.pth",
        "class_map.json",
        "pyproject.toml",
        "README.md",
    ]
    
    # Copy main files
    for file in files_to_copy:
        src = root / file
        if src.exists():
            shutil.copy2(src, package_dir / file)
            print(f"✓ Copied {file}")
        else:
            print(f"✗ Warning: {file} not found")
    
    # Copy src folder (only necessary files for inference)
    src_dir = root / "src"
    dst_src_dir = package_dir / "src"
    dst_src_dir.mkdir()
    
    inference_files = [
        "__init__.py",
        "model.py",
        "inference.py",
        "gradcam.py",
        "dataset.py",
        "memory.py",
        "pixiv_downloader.py",
    ]
    
    for file in inference_files:
        src = src_dir / file
        if src.exists():
            shutil.copy2(src, dst_src_dir / file)
            print(f"✓ Copied src/{file}")
    
    # Copy release files (bat and requirements only)
    release_files_dir = root / "release"
    if release_files_dir.exists():
        for item in release_files_dir.iterdir():
            if item.is_file() and item.suffix in ['.bat', '.txt']:
                shutil.copy2(item, package_dir / item.name)
                print(f"✓ Copied release/{item.name}")
    
    # Create zip file
    zip_path = release_dir / f"{package_name}.zip"
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for file_path in package_dir.rglob("*"):
            if file_path.is_file():
                arcname = file_path.relative_to(release_dir)
                zipf.write(file_path, arcname)
    
    print(f"\n{'='*50}")
    print(f"✓ Release package created: {zip_path}")
    print(f"  Size: {zip_path.stat().st_size / (1024*1024):.1f} MB")
    print(f"{'='*50}")
    
    return zip_path

if __name__ == "__main__":
    create_release_package()
