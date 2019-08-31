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
        scFiltered.setup(env, StreamCompaction::CompType::Digit, 1, digitnum);
        worker         = sc.worker();
        workerFiltered = scFiltered.worker();
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

    void compactFilteredPeaks(ClusterFinderState &state, cl::CommandQueue queue)
    {
        state.filteredPeakNum = workerFiltered->run(
                state.peaknum,
                queue,
                state.peaks,
                state.filteredPeaks,
                state.isPeak);
    }

    Step step()
    {
        return worker->asStep("compactPeaks");
    }

    Step stepFiltered()
    {
        return workerFiltered->asStep("compactFilteredPeaks");
    }
    
private:

    StreamCompaction sc;    
    nonstd::optional<StreamCompaction::Worker> worker;

    StreamCompaction scFiltered;    
    nonstd::optional<StreamCompaction::Worker> workerFiltered;

};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
