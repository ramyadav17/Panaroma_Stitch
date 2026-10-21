# Panorama Lab

An interactive Flask front end for the panorama-stitching pipeline in `Panorama_Stitching_Project.ipynb`.

## Run

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Open `http://127.0.0.1:5000`, upload 2–6 overlapping images in left-to-right order, choose SIFT or ORB and a blend mode, then create and download the panorama.

The backend in `panorama.py` retains the notebook's feature matching, DLT homography, custom RANSAC, inverse bilinear warp, and feather blending. Uploads are resized to a 1400 px maximum dimension for responsive interaction.
