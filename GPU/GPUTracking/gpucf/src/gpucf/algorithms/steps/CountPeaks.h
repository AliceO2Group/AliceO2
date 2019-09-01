#pragma once
        
#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class CountPeaks : public Kernel1D
{

public:
    DECL_KERNEL(CountPeaks, "countPeaks");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        bool scratchpad = (state.cfg.clusterbuilder == ClusterBuilder::ScratchPad);
        size_t dummyItems = (scratchpad) ? 64 - (state.digitnum % 64) : 0;

        size_t workitems = state.digitnum + dummyItems;

        kernel.setArg(0, state.peakMap);
        kernel.setArg(1, state.chargeMap);
        kernel.setArg(2, state.digits);
        kernel.setArg(3, static_cast<cl_uint>(state.digitnum));

        Kernel1D::call(0, workitems, 64, queue);
    }

};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:

