#pragma once

#include <gpucf/algorithms/ClusterFinderState.h>
#include <gpucf/algorithms/StreamCompaction.h>

#include <nonstd/optional.hpp>


namespace gpucf
{

class CompactPeaks
{

public:

    CompactPeaks(ClEnv env, size_t digitnum)
    {
        sc.setup(env, StreamCompaction::CompType::Digit, 1, digitnum);
        worker = sc.worker();
    }

    void call(ClusterFinderState &state, cl::CommandQueue queue)
    {
        state.peaknum = worker->run(
                state.digitnum, 
                queue, 
                state.digits, 
                state.peaks, 
                state.isPeak);
    }

    Step step()
    {
        return worker->asStep("compactPeaks");
    }
    
private:

    StreamCompaction sc;    
    nonstd::optional<StreamCompaction::Worker> worker;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
