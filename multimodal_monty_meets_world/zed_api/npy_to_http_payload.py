import numpy as np
import argparse
import os

''' Example usage:
For default output path (depth_x.data):
$ python npy_to_http_payload.py path/to/depth.npy

To specify output path and read centre depth value:
$ python npy_to_http_payload.py path/to/depth.npy --output_path path/to/output.data --centre
'''

def convert_npy_to_http_payload(npy_file_path, output_filename="depth_x.data"):
    """
    Convert a depth .npy map to HTTP payload (raw bytes) format.
    
    The output is a binary file containing the depth values as raw bytes, with no header or metadata. The depth values are stored as 32-bit floats.
    - HTTP payload written as raw bytes with no header
    - Format: 32-bit float
    - Layout: flat 1D array of length width * height
    - Units: metres
    """
    try:
        print(f"Processing NPY: {npy_file_path}")
        
        # 1. Load the NPY file
        depth_array = np.load(npy_file_path)
        if depth_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D depth array in .npy, got shape {depth_array.shape}"
            )
        height, width = depth_array.shape
        
        # 2. Convert to float32 if needed
        if depth_array.dtype == np.float16:
            depth_float32 = depth_array.astype(np.float32)
            print(f"Converted Float16 to Float32")
        elif depth_array.dtype in [np.uint8, np.uint16, np.uint32]:
            # If integer data, convert to float32
            depth_float32 = depth_array.astype(np.float32)
            print(f"Converted {depth_array.dtype} to Float32")
        else:
            # Already float or other format
            depth_float32 = depth_array.astype(np.float32)
        
        # 3. Flatten to 1D array (width * height)
        flat_payload = depth_float32.flatten()
        
        # Verify dimensions
        expected_size = width * height
        assert len(flat_payload) == expected_size, \
            f"Size mismatch: expected {expected_size}, got {len(flat_payload)}"
        
        # 4. Save as raw bytes (no header)
        raw_bytes = flat_payload.tobytes()
        
        with open(output_filename, "wb") as f:
            f.write(raw_bytes)
        
        print(f"  Saved {len(raw_bytes)} bytes ({width}x{height}) to '{output_filename}'")
        print(f"  Data range: {flat_payload.min():.6f} to {flat_payload.max():.6f}")
        print(f"  Units: metres (assumed)")
        print(f"  File saved to: {os.path.abspath(output_filename)}")
        
        return raw_bytes, width, height
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None


def read_centre_depth(npy_file_path):
    """
    Read the depth value at the centre of the NPY depth map.
    Useful for verifying if the depth value is as expected (in metres).
    """
    try:
        print(f"\nReading centre value from: {npy_file_path}")
        
        # 1. Load the NPY file
        depth_array = np.load(npy_file_path)
        if depth_array.ndim != 2:
            raise ValueError(
                f"Expected a 2D depth array in .npy, got shape {depth_array.shape}"
            )
        height, width = depth_array.shape
        
        # 2. Convert to float32 if needed
        depth_float32 = depth_array.astype(np.float32)
        
        # 3. Calculate centre coordinates
        centre_x = width // 2
        centre_y = height // 2
        
        # 4. Read centre pixel value
        centre_value = depth_float32[centre_y, centre_x]
        
        print(f"  Image size: {width}x{height}")
        print(f"  centre position: ({centre_x}, {centre_y})")
        print(f"  centre depth value: {centre_value} metres")
        print(f"  Data range: {depth_float32.min():.6f} to {depth_float32.max():.6f} metres\n")
        
        return centre_value
        
    except Exception as e:
        print(f"Error reading centre value: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert NPY depth map to HTTP payload (raw bytes)")
    parser.add_argument("npy_path", help="Path to the NPY depth map file")
    parser.add_argument("--output_path", default="depth_x.data", help="Path to save the output binary file (default: depth_x.data)")
    parser.add_argument("--centre", action="store_true", help="Read and display centre depth value")
    
    args = parser.parse_args()
    
    # Read centre value to verify depth if requested
    if args.centre:
        read_centre_depth(args.npy_path)
    
    # Convert and save
    convert_npy_to_http_payload(args.npy_path, args.output_path)