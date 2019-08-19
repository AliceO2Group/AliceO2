#pragma once

#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class ResetMaps : public Kernel1D
{

public:
    DECL_KERNEL(ResetMaps, "resetMaps");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        kernel.setArg(0, state.digits);
        kernel.setArg(1, state.chargeMap);
        kernel.setArg(2, state.peakCountMap);
        kernel.setArg(3, state.isPeak);
        Kernel1D::call(0, state.digitnum, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

