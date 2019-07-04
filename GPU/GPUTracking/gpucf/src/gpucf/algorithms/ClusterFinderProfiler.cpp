#include "ClusterFinderProfiler.h"


using namespace gpucf;


std::vector<Step> ClusterFinderProfiler::run(nonstd::span<const Digit> digits)
{
    digitsToGPU.call(state, digits, queue);

    fillChargeMap.call(state, queue);

    findPeaks.call(state, queue);

    compactPeaks.call(state, queue);
    
    countPeaks.call(state, queue);

    computeCluster.call(state, queue);

    compactCluster.call(state, queue);

    queue.finish();


    std::vector<Step> steps = {
        fillChargeMap,
        findPeaks,
        countPeaks,
        compactPeaks.step(),
        computeCluster,
        compactCluster.step(),
        resetMaps,
    };

    for (Step &step : steps)
    {
        step.lane = 0;
    }

    return steps;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
