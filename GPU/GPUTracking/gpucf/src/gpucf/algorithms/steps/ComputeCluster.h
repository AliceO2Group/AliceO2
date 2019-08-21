#pragma once

#include <gpucf/common/Kernel1D.h>


namespace gpucf
{

class ComputeCluster : public Kernel1D
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
        kernel.setArg(2, cl_uint(state.peaknum));
        kernel.setArg(3, cl_uint(state.maxClusterPerRow));
        kernel.setArg(4, state.clusterInRow);
        kernel.setArg(5, state.clusterByRow);

        Kernel1D::call(0, workitems, 64, queue);
    }
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
