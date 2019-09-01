#include "ClusterFinderProfiler.h"


using namespace gpucf;


std::vector<Step> ClusterFinderProfiler::run(nonstd::span<const Digit> digits)
{
    digitsToGPU.call(state, digits, queue);

    fillChargeMap.call(state, queue);

    findPeaks.call(state, queue);

    compactPeaks.call(state, queue);

    noiseSuppression.call(state, queue);

    compactPeaks.compactFilteredPeaks(state, queue);

    log::Debug() << "Counting Peaks START";
    countPeaks.call(state, queue);
    log::Debug() << "Counting Peaks END";

    log::Debug() << "Computing cluster START";
    computeCluster.call(state, queue);
    log::Debug() << "Computing cluster END";

    log::Debug() << "Reset maps START";
    resetMaps.call(state, queue);
    log::Debug() << "Reset maps END";

    log::Debug() << "Finish queue START";
    queue.finish();
    log::Debug() << "Finish queue END";


    log::Info() << "Collecting OpenCL profiling data.";
    std::vector<Step> steps = {
        fillChargeMap,
        findPeaks,
        compactPeaks.step(),
        noiseSuppression.asStep("noiseSuppression"),
        compactPeaks.stepFiltered(),
        countPeaks,
        computeCluster,
        resetMaps
    };

    for (Step &step : steps)
    {
        step.lane = 0;
    }

    return steps;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
