#pragma once

#include <gpucf/algorithms/ClusterFinderState.h>
#include <gpucf/algorithms/StreamCompaction.h>

#include <nonstd/optional.hpp>


namespace gpucf
{

class CompactClusters
{

public:
    CompactClusters(ClEnv env, size_t digitnum)
    {
        sc.setup(env, StreamCompaction::CompType::Cluster, 1, digitnum);
        worker = sc.worker();
    }

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        state.cutoffClusternum = worker->run(
                state.peaknum, 
                queue,
                state.clusterNative, 
                state.clusterNativeCutoff, 
                state.aboveQTotCutoff);
    }
    

private:
    StreamCompaction sc;    
    nonstd::optional<StreamCompaction::Worker> worker;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
