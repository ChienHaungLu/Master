"""
Stain Normalization Script for Pathology Images
Uses Macenko method to normalize H&E stain colors
"""

import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# ============================================
# Macenko Stain Normalization Implementation
# ============================================

class MacenkoNormalizer:
    """
    Stain normalization using Macenko's method
    Reference: A method for normalizing histology slides for quantitative analysis
    """
    
    def __init__(self):
        self.HERef = np.array([[0.5626, 0.2159],
                               [0.7201, 0.8012],
                               [0.4062, 0.5581]])
        self.maxCRef = np.array([1.9705, 1.0308])
        
    def __convert_rgb_to_od(self, img):
        """Convert RGB to Optical Density"""
        img = img.astype(np.float64) + 1
        return -np.log(img / 256)
    
    def __convert_od_to_rgb(self, od):
        """Convert Optical Density to RGB"""
        return (256 * np.exp(-od) - 1).astype(np.uint8)
    
    def __find_he_components(self, od, percentile=99):
        """Find H&E stain vectors using SVD"""
        # Remove low optical density pixels
        od_hat = od[~np.any(od < 0.15, axis=1)]
        
        if od_hat.shape[0] < 10:
            return None, None
        
        # SVD
        _, _, V = np.linalg.svd(od_hat, full_matrices=False)
        
        # Project onto plane
        plane = V[:2, :].T
        proj = np.dot(od_hat, plane)
        
        # Find angles
        phi = np.arctan2(proj[:, 1], proj[:, 0])
        
        min_phi = np.percentile(phi, 100 - percentile)
        max_phi = np.percentile(phi, percentile)
        
        # Get stain vectors
        v1 = np.array([np.cos(min_phi), np.sin(min_phi)])
        v2 = np.array([np.cos(max_phi), np.sin(max_phi)])
        
        he = np.array([np.dot(plane, v1), np.dot(plane, v2)]).T
        
        # Make sure H comes before E
        if he[0, 0] < he[0, 1]:
            he = he[:, [1, 0]]
            
        return he, od_hat
    
    def fit(self, target_img):
        """Fit normalizer to target (reference) image"""
        img = np.array(target_img)
        od = self.__convert_rgb_to_od(img.reshape(-1, 3))
        
        he, _ = self.__find_he_components(od)
        if he is None:
            print("Warning: Could not fit reference image")
            return False
            
        self.HERef = he
        
        # Get max concentrations
        y = np.linalg.lstsq(he, od.T, rcond=None)[0]
        self.maxCRef = np.percentile(y, 99, axis=1)
        
        return True
    
    def transform(self, source_img):
        """Transform source image to match reference stain"""
        img = np.array(source_img)
        h, w, _ = img.shape
        
        od = self.__convert_rgb_to_od(img.reshape(-1, 3))
        
        he, _ = self.__find_he_components(od)
        if he is None:
            return source_img  # Return original if normalization fails
        
        # Get stain concentrations
        y = np.linalg.lstsq(he, od.T, rcond=None)[0]
        
        # Normalize
        max_c = np.percentile(y, 99, axis=1, keepdims=True)
        max_c[max_c == 0] = 1  # Avoid division by zero
        y = y / max_c * self.maxCRef[:, np.newaxis]
        
        # Reconstruct
        od_normalized = np.dot(self.HERef, y).T
        img_normalized = self.__convert_od_to_rgb(od_normalized)
        
        return Image.fromarray(img_normalized.reshape(h, w, 3))


def normalize_dataset(source_dir, target_dir, reference_img_path):
    """
    Normalize all images in source_dir using reference image
    
    Args:
        source_dir: Path to source images
        target_dir: Path to save normalized images
        reference_img_path: Path to reference image for normalization
    """
    source_path = Path(source_dir)
    target_path = Path(target_dir)
    target_path.mkdir(parents=True, exist_ok=True)
    
    # Initialize normalizer with reference image
    normalizer = MacenkoNormalizer()
    ref_img = Image.open(reference_img_path).convert('RGB')
    
    if not normalizer.fit(ref_img):
        print("Error: Failed to fit reference image")
        return
    
    print(f"Reference image fitted: {reference_img_path}")
    
    # Get all images
    valid_exts = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
    image_files = [f for f in source_path.iterdir() 
                   if f.is_file() and f.suffix.lower() in valid_exts]
    
    print(f"Found {len(image_files)} images to normalize")
    
    success_count = 0
    fail_count = 0
    
    for img_file in tqdm(image_files, desc="Normalizing"):
        try:
            img = Image.open(img_file).convert('RGB')
            normalized = normalizer.transform(img)
            
            # Save with same filename
            save_path = target_path / img_file.name
            normalized.save(save_path)
            success_count += 1
            
        except Exception as e:
            print(f"\nWarning: Failed to normalize {img_file.name}: {e}")
            # Copy original if normalization fails
            img = Image.open(img_file).convert('RGB')
            img.save(target_path / img_file.name)
            fail_count += 1
    
    print(f"\nDone! Success: {success_count}, Failed (copied original): {fail_count}")


if __name__ == "__main__":
    from pathlib import Path
    
    ROOT = Path("/home/liang/workdir/NCKU-ML-25Fall/Final_Project")
    
    # Source directories
    POS_SOURCE = ROOT / "raw" / "positive"
    NEG_SOURCE = ROOT / "raw" / "negative"
    
    # Target directories
    POS_TARGET = ROOT / "raw_normalized" / "positive"
    NEG_TARGET = ROOT / "raw_normalized" / "negative"
    
    # Use first positive image as reference (you can change this)
    # Find first image
    pos_images = list(POS_SOURCE.glob("*.png")) + list(POS_SOURCE.glob("*.jpg"))
    REFERENCE_IMG = pos_images[0] if pos_images else None
    
    if REFERENCE_IMG is None:
        print("Error: No images found in positive directory")
        exit(1)
    
    print(f"Using reference image: {REFERENCE_IMG}")
    print("=" * 50)
    
    print("\n[1/2] Normalizing POSITIVE images...")
    normalize_dataset(POS_SOURCE, POS_TARGET, REFERENCE_IMG)
    
    print("\n[2/2] Normalizing NEGATIVE images...")
    normalize_dataset(NEG_SOURCE, NEG_TARGET, REFERENCE_IMG)
    
    print("\n" + "=" * 50)
    print("All done! Normalized images saved to:")
    print(f"  - {POS_TARGET}")
    print(f"  - {NEG_TARGET}")
