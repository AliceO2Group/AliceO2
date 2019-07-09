#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/ClEnv.h>

#include <CL/cl2.hpp>


namespace gpucf
{
    
struct ClusterFinderState
{
    ClusterFinderConfig cfg;
    
    size_t digitnum;
    cl::Buffer digits;

    cl::Buffer isPeak;

    size_t peaknum;
    cl::Buffer peaks;

    cl::Buffer chargeMap;
    cl::Buffer peakMap;
    cl::Buffer peakCountMap;
    
    size_t maxClusterPerRow;
    cl::Buffer clusterInRow;
    cl::Buffer clusterByRow;

    ClusterFinderState(ClusterFinderConfig, size_t, cl::Context, cl::Device);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
