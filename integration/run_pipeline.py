import subprocess
import os

BASE = "/home/sra/freeze"

GEDI_ENV   = f"{BASE}/gedi_test/gedi/.venv/bin/activate"
GEDI_SCRIPT = f"{BASE}/gedi_test/gedi/1_gedi_process_object.py"

DINO_ENV   = f"{BASE}/dino_test/dinov2/.venv/bin/activate"
DINO_SCRIPT = f"{BASE}/dino_test/dinov2/demo_infer.py"  

# -------------------------------------------------------------
def run_gedi(object_folder, output_folder):
    """
    Runs GeDi on an object folder:
      object_folder = "data/20objects/data/Kinfu_Audiobox1_light"
    """
    cmd = (
        f"source {GEDI_ENV} && "
        f"python {GEDI_SCRIPT} {object_folder} {output_folder}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
def run_dino(image_path, output_path):
    """
    Runs DINOv2 on an image.
    """
    cmd = (
        f"source {DINO_ENV} && "
        f"python {DINO_SCRIPT} {image_path} {output_path}"
    )
    subprocess.run(["bash", "-c", cmd], check=True)

# -------------------------------------------------------------
if __name__ == "__main__":

    # geometric data folder for one object
    obj_folder = f"{BASE}/data/20objects/data/Kinfu_Audiobox1_light"

    # where to store output
    out_folder = f"{BASE}/processed"
    os.makedirs(out_folder, exist_ok=True)

    print("Running GeDi preprocessing & descriptor extraction...")
    run_gedi(obj_folder, out_folder)

    # run DINO on one image from the same folder
    dino_in = f"{obj_folder}/rgb_noseg/color_00008.png"
    dino_out = f"{out_folder}/Kinfu_Audiobox1_light_dino.npy"

    print("Running DINOv2 feature extraction...")
    run_dino(dino_in, dino_out)

    print("Done.")
