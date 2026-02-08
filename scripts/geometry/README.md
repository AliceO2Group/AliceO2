This is the tool O2_CADtoTGeo.py which translates from geometries in STEP format (CAD export) to
TGeo.

To use the tool, setup a conda environment with python-occ core installed.
The following should work on standard linux x86:

```
# -) download miniconda into $HOME/miniconda (if not already done)
if [ ! -d $HOME/miniconda ]; then
  curl -fsSL https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -o miniconda.sh
  bash miniconda.sh -b -p $HOME/miniconda
fi

# -) source conda into the environment (in every shell you want to use this)
source $HOME/miniconda/etc/profile.d/conda.sh

# -) Create an OCC environment (for OpenCacade)
conda create -n occ python=3.10 -y
conda activate occ

# 3) Install OpenCascade Python bindings
conda install -c conda-forge pythonocc-core -y

# 4) Run the tool, e.g.
conda activate occ
python PATH_TO_ALICEO2_SOURCES/scripts/geometry/O2_CADtoTGeo.py --help
```