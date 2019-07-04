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
    size_t peaknum;
    size_t cutoffClusternum;

    cl::Buffer digits;
    cl::Buffer isPeak;
    cl::Buffer peaks;
    cl::Buffer chargeMap;
    cl::Buffer peakMap;
    cl::Buffer peakCountMap;
    
    cl::Buffer aboveQTotCutoff;
    cl::Buffer clusterNative;
    cl::Buffer rows;
    cl::Buffer clusterNativeCutoff;

    cl::Buffer globalToLocalRow;
    cl::Buffer globalRowToCru;

    ClusterFinderState(ClusterFinderConfig, size_t, cl::Context, cl::Device);
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
