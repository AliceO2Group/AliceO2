#pragma once

#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class ComputeCluster : protected Kernel1D
{

public:
    DECL_KERNEL(ComputeCluster, "computeClusters");

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        bool scratchpad = (state.cfg.clusterbuilder == ClusterBuilder::ScratchPad);
        size_t dummyItems = (scratchpad) ?  64 - (state.peaknum % 64) : 0;
        size_t workitems = state.peaknum + dummyItems;

        kernel.setArg(0, state.chargeMap);
        kernel.setArg(1, state.peaks);
        kernel.setArg(2, state.globalToLocalRow);
        kernel.setArg(3, state.globalRowToCru);
        kernel.setArg(4, cl_uint(state.peaknum));
        kernel.setArg(5, state.clusterNative);
        kernel.setArg(6, state.aboveQTotCutoff);
        kernel.setArg(7, state.peakMap);

        Kernel1D::call(0, workitems, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
