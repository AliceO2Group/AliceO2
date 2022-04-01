<!-- doxy
\page refGPUTrackingStandalone Standalone benchmark
/doxy -->

# GPU Tracking Standalone benchmark

This is the standalone benchmark for the GPU tracking.

In order to build: Use the CMake in this folder. The build can be configured using config.cmake. By default, the cmake process will link the src folder as symlink src in the installation folder, as well as the config.cmake, and provide a makefile there to trigger a rebuild and reinstall.
In order to have all dependencies ready for the standalone build:
- Either have them all as system packages.
- Or build O2 with aliBuild, and then you can use the O2 dependencies by sourcing O2/GPU/GPUTracking/Standalone/cmake/prepare.sh from your alice folder (or set `ALIBUILD_WORK_DIR` accordingly)

Afterwards, you can build the standalone benchmark in that shell:
```
mkdir -p standalone/build
cd standalone/build
cmake -DCMAKE_INSTALL_PREFIX=../ $ALIBUILD_WORK_DIR/../O2/GPU/GPUTracking/Standalone
make -j install
```

In order to change cmake settings and rebuild, do in the `standalone` folder:
```
nano config.cmake #edit as you wish
make -j # will rebuild and reinstall into this folder
```

In order to run, run `./ca` in the standalone folder. `./ca --helpall` will show all command line options.

In order to extract data dumps from O2 to be run in the standalone benchmark:
- Run the `o2-gpu-reco-workflow` with `--configKeyValues="GPU_global.dump=1;"`.
- move all the created `*.dump` files to `standalone/events/[some_name]`.
- Run `./ca -e [some_name]`.
