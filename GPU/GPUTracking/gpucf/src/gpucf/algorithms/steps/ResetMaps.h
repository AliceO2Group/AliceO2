#pragma once

#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class ResetMaps : protected Kernel1D
{

public:
    DECL_KERNEL(ResetMaps, "resetMaps");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        kernel.setArg(0, state.chargeMap);
        kernel.setArg(1, state.peaks);
        Kernel1D::call(0, state.digitnum, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

