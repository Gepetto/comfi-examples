## Data Organization

Data from Zenodo (https://zenodo.org/records/17223909) should be downloaded and organized as shown in the following diagram:

![COMFI Organization](images/comfi_organisation.pdf)

In particular, we recommend users to extract all the videos zip files into a same videos folder located into COMFI and to put the whole dataset into the present repository.

# comfi-usage

Scripts to use and showcase the data from the COMFI dataset

## Installation

1. **Clone the repository**

```bash
git clone https://github.com/Gepetto/comfi-examples.git
cd comfi-usage
```
2. **Virtual environment**
```bash
python3 -m venv comfi_env
source comfi_env/bin/activate
```

3. **Install dependencies**
```bash
pip install --upgrade pip
pip install -r requirements.txt
```
### Usage

This repository provides scripts for processing and visualizing COMFI dataset data. Visualizations are performed using Meshcat, which by default launches a local server (typically at http://127.0.0.1:7000
) to stream the visualizations directly in your browser.

All data can be visualized and processed by specifying the subject ID, task, and other options available for each script (see python script.py --help for details). All the commands set to launch the code examples must be used from the repository root.

1. **Human pose estimation (HPE)**

Run the RTMlib pose estimator (YOLOX + RTMPose-M with 26 keypoints), which can display the results in real time (this option can be disabled). At the end of the run, an output video with the skeleton overlay and a CSV file containing both the average score and the keypoints are saved in the output folder.

```bash
python scripts/human_pose_estimator/run_pose_estimator.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root ./COMFI
```
2. **Triangulation and visualization**

Triangulate keypoints from multiple camera views (can be done with any set of cameras, minimum 2). The triangulation results are saved in the output folder.
```bash
python scripts/run_triangulation.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root ./COMFI
```
2.1 **Visualize triangulated data:**
```bash
python scripts/visualization/viz_jcp.py \
    --mode hpe \
    --id 4279 \
    --task RobotWelding \
    --nb-cams 4 \
    --freq 40 \
    --start 0 \
    --stop 500
```
3. **Visualize mocap data (markers and joint center positions"JCP") at either 40 Hz or 100 Hz.**
```bash
python scripts/visualization/viz_mks.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root ./COMFI \
    --freq 40 \
    --mkset est \
    --with_jcp \
    --start 100 \
    --stop 500
```
4. **Visualize all_data**

Visualize multimodal data and animate motion capture sequences, including reference 3D marker positions, JCP, resultant force vectors, and the poses of the world, cameras, force plates, and robot frames. The animation shows both the biomechanical model, built from the scaled URDF, and the robot’s motion. Optionally, JCP from HPE or aligned data can also be visualized if --with-jcp-hpe is set to true.

**Note:** Robot and force data are not available for all tasks. Additionally, robot data is only aligned with videos at 40 Hz.
```bash
python scripts/visualization/viz_all_data.py \
     --id 1012 \
     --task RobotWelding \
     --comfi-root ./COMFI \
     --freq 40 \
     --start 100 \
     --with-jcp-hpe \ (optional)
     --jcp-hpe-mode aligned

```
5. **Extract joint center positions from mocap markers (using our markerset)**

The extracted JCP are saved in the output folder.
```bash
python scripts/get_jcp_from_mocap_markers.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root ./COMFI \
    --freq 40 \
    --mkset est

```
6. **Procruste alignment using the Kabsch algorithm**

Performs a Procrustes alignment between JCP Mocap and JCP HPE. The newly aligned JCP are saved in the output folder.
```bash
python scripts/run_procruste_alignement.py \
    --id 1012 \
    --task RobotWelding \
    --comfi-root ./COMFI \
    --nb-cams 4
```
