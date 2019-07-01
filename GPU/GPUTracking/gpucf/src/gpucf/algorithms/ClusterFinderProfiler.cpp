#include "ClusterFinderProfiler.h"


using namespace gpucf;


std::vector<Step> ClusterFinderProfiler::run(nonstd::span<const Digit> digits)
{
    digitsToGPU.call(state, digits, queue);

    fillChargeMap.call(state, queue);

    findPeaks.call(state, queue);

    compactPeaks.call(state, queue);
    
    if (state.cfg.splitCharges)
    {
        countPeaks.call(state, queue);
    }

    computeCluster.call(state, queue);

    compactCluster.call(state, queue);

    queue.finish();


    std::vector<Step> steps = {
        fillChargeMap,
        findPeaks,
        (state.cfg.splitCharges) 
            ? Step{countPeaks}
            : Step{"countPeaks", 0, 0, 0, 0},
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
