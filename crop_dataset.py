#!/usr/bin/env python
"""
Crop and georeference radar prediction TIFF files.

This script scans a source directory for TIFF files (``.tif`` or ``.tiff``),
assigns a spatial reference system (EPSG:4326) and a geotransform based on
the bounding box of Italy, and writes the results to a destination
directory named ``predictions_cropped``.  If a file with the same name
already exists in the destination directory, a numeric suffix is appended
to avoid overwriting.

The goal is to produce GeoTIFF files ready for alignment with DEM and
land-cover data used in flood modelling.  The default bounding box
matches the coordinates used throughout this project (min_lon,
max_lon, min_lat, max_lat).  Adjust the constants at the top of the
file if your area of interest differs.

Usage:
    python crop_move_predictions.py

You can also import the functions in this module and call
``crop_and_georeference`` or ``process_directory`` directly with custom
paths or bounding boxes.
"""

import os
import rasterio
from rasterio.transform import from_bounds
from typing import Iterable


# Default bounding box for Italy (longitude/latitude in degrees)
ITALY_BBOX = {
    "min_x": 6.7499552751,
    "max_x": 18.4802470232,
    "min_y": 36.619987291,
    "max_y": 47.1153931748,
}


def crop_and_georeference(src_path: str, dest_path: str, bbox: dict = ITALY_BBOX) -> None:
    """Assign EPSG:4326 CRS and geotransform to a TIFF file.

    Parameters
    ----------
    src_path : str
        Path to the source TIFF file.
    dest_path : str
        Path to write the processed TIFF.
    bbox : dict, optional
        Bounding box with keys ``min_x``, ``max_x``, ``min_y``, ``max_y``.

    Notes
    -----
    This function does not crop pixel data; instead it sets the geotransform
    so that the full image covers the specified bounding box.  This
    effectively "georeferences" the image in EPSG:4326.  If cropping
    to a subset of the bounding box is required, additional logic
    should be implemented here.
    """
    with rasterio.open(src_path) as src:
        # Read the first band
        data = src.read(1)
        height, width = data.shape

        # Compute the new affine transform based on the bounding box
        transform = from_bounds(
            float(bbox["min_x"]), float(bbox["min_y"]),
            float(bbox["max_x"]), float(bbox["max_y"]),
            width, height
        )

        # Build a new profile for output
        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": height,
            "width": width,
            "count": 1,
            "crs": "EPSG:4326",
            "transform": transform,
        })

        # Write out the georeferenced file
        with rasterio.open(dest_path, "w", **profile) as dst:
            dst.write(data, 1)


def process_directory(src_dir: str, dest_dir: str, extensions: Iterable[str] = (".tif", ".tiff")) -> None:
    """Process all TIFF files in ``src_dir`` and save results to ``dest_dir``.

    Parameters
    ----------
    src_dir : str
        Directory containing source TIFF files.
    dest_dir : str
        Directory to write processed TIFF files.  Created if missing.
    extensions : iterable of str, optional
        File extensions to consider as TIFF files (case-insensitive).
    """
    # Ensure source exists
    if not os.path.isdir(src_dir):
        raise FileNotFoundError(f"Source directory does not exist: {src_dir}")

    # Create destination directory if necessary
    os.makedirs(dest_dir, exist_ok=True)

    # Normalize extension list
    ext_set = tuple(ext.lower() for ext in extensions)

    # Iterate over items in source directory
    for filename in os.listdir(src_dir):
        src_path = os.path.join(src_dir, filename)
        if not os.path.isfile(src_path):
            continue
        ext = os.path.splitext(filename)[1].lower()
        if ext not in ext_set:
            continue

        # Prepare destination path (handle duplicates)
        dest_path = os.path.join(dest_dir, filename)
        if os.path.exists(dest_path):
            base, extension = os.path.splitext(filename)
            counter = 1
            while True:
                new_filename = f"{base}_{counter}{extension}"
                new_path = os.path.join(dest_dir, new_filename)
                if not os.path.exists(new_path):
                    dest_path = new_path
                    break
                counter += 1

        # Process file
        crop_and_georeference(src_path, dest_path)
        print(f"Processed {src_path} -> {dest_path}")


def main() -> None:
    """Entry point for command-line execution.

    This uses the hard-coded directories for the radar predictions used in
    the Vincenzo project.  Adjust ``src_dir`` and ``dest_dir`` here if
    your environment differs.
    """
    src_dir = "/home/v.bucciero/data/instruments/rdr0_previews_h100gpu/epoch_000/predictions"
    dest_dir = os.path.join(os.path.dirname(src_dir), "predictions_cropped")
    process_directory(src_dir, dest_dir)


if __name__ == "__main__":
    main()