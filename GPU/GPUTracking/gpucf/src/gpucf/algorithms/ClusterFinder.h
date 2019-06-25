#pragma once

#include <gpucf/algorithms/steps/steps.h>

#include <CL/cl2.hpp>


namespace gpucf
{

class ClusterFinder
{

public:

    ClusterFinder(ClusterFinderConfig, size_t, ClEnv);

protected:

    cl::CommandQueue queue;

    ClusterFinderState state;

    CompactClusters compactCluster;
    CompactPeaks    compactPeaks;
    ComputeCluster  computeCluster;
    CountPeaks      countPeaks;
    DigitsToGPU     digitsToGPU;
    FillChargeMap   fillChargeMap;
    FindPeaks       findPeaks;
    NativeToRegular nativeToRegular;
    ResetMaps       resetMaps;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
