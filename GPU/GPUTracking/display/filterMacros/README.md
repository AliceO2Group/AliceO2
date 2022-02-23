<!-- doxy
\page refGPUTrackingDisplayFilterMacros Filtering tracks macros
/doxy -->

# Filtering tracks in the GPU event display using ROOT macros

The event display can filter tracks using ROOT macros.
For the standalone version, you can provide a list of filter macros with the `--GLfilterMacros` option.
In the DPL version, you can set a single filter macro via `--configKeyValues "GPU_global.gpuDisplayfilterMacro=[MACRO]"`.
By pressing the `u` key in the event display, you can cycle through the loaded filters (and filter disabled).
You can use the ROOT `+` and `++` syntax for compiled macros.
All macro files must be placed in the folder `displayTrackFilter` in the working directory.
The macros should include `#include "GPUO2Interface.h"`

The filter macro must implement a function with the following prototype:
```
void gpuDisplayTrackFilter(std::vector<bool>* filter, const GPUTrackingInOutPointers* ioPtrs, const GPUConstantMem* processors)
```
This function is called once when the event display loads the data for an event / time frame.
Here, `filter` is a vector of booleans indicating whether the track is shown or not. Default is all entries are `true` and no track is filtered.
The current filter applies only to TPC tracks, and will filter matched ITS, TOF, and TRD tracks as well.
It is not yet possible to filter ITS standalone tracks.
Access to the track data goes via the `ioPtrs` structure.
Note that when the input to the event display is GPU tracks, use `ioPtrs->mergedTracks` while when the input are final TPC tracks, `ioPtrs->outputTracksTPCO2` must be used.

Note that since most GPU classes have no ROOT dictionary, it might be required to use the macro in compiled mode.

If the event display runs with the tracking (i.e. not via the separate DPL device but in the same executable as the tracking), the filter has access to all internal data structures of the tracker via the `processors` parameter.

Some example filter macros are placed in O2 in `GPU/GPUTracking/display/filterMacros`.
