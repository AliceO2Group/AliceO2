#pragma once

#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class NativeToRegular : protected Kernel1D
{

public:
    DECL_KERNEL(NativeToRegular, "nativeToRegular");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        kernel.setArg(0, state.clusterNativeCutoff);
        kernel.setArg(1, state.peaks);
        kernel.setArg(2, state.globalToLocalRow);
        kernel.setArg(3, state.globalRowToCru);
        kernel.setArg(4, state.cluster);

        Kernel1D::call(0, state.peaknum, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
