#include "ClusterFinderTest.h"


using namespace gpucf;


void ClusterFinderTest::run(nonstd::span<const Digit> digits)
{
    digitsToGPU.call(state, digits, queue);

    fillChargeMap.call(state, queue);
    findPeaks.call(state, queue);
    compactPeaks.call(state, queue);

    if (state.cfg.splitCharges)
    {
        countPeaks.call(state, queue);
    }

    compactCluster.call(state, queue);

    nativeToRegular.call(state, queue);

    queue.finish();
}

// vim: set ts=4 sw=4 sts=4 expandtab:
