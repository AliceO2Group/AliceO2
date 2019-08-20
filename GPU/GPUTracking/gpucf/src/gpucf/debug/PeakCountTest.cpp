#include "PeakCountTest.h"


using namespace gpucf;


PeakCountTest::PeakCountTest(ClusterFinderConfig config, ClEnv env)
    : ClusterFinder(config, 100, env)
{
}

bool PeakCountTest::run(
        const Array2D<float> &charges,
        const Array2D<unsigned char> &/*isPeakGT*/,
        const Array2D<char> &/*peakCountGT*/)
{
    std::vector<Digit> digits = digitize(charges);

    for (const Digit &d : digits)
    {
        log::Debug() << d;
    }

    digitsToGPU.call(state, digits, queue);

    fillChargeMap.call(state, queue);

    findPeaks.call(state, queue);

    countPeaks.call(state, queue);

    size_t timebins = getWidthTime(charges);
    size_t pads = getWidthPad(charges);
    size_t elems = TPC_NUM_OF_PADS * (timebins + PADDING);

    std::vector<unsigned short> chargeMapBuf(elems);
    gpucpy<unsigned short>(
            state.chargeMap,
            chargeMapBuf,
            chargeMapBuf.size(),
            queue, 
            true);
    Map<unsigned short> chargeMap = mapify<unsigned short>(chargeMapBuf, 0, pads, timebins);

    std::vector<unsigned char> isPeakBuf(elems);
    gpucpy<unsigned char>(state.peakMap, isPeakBuf, isPeakBuf.size(), queue, true);
    Map<unsigned char> isPeak = mapify<unsigned char>(isPeakBuf, 0, pads, timebins);

    std::vector<char> peakCountBuf(elems);
    gpucpy<char>(
            state.peakCountMap, 
            peakCountBuf,
            peakCountBuf.size(), 
            queue,
            true);
    Map<char> peakCount = mapify<char>(peakCountBuf, 1, pads, timebins);

    log::Debug() << "chargeMap\n" << print(chargeMap, pads, timebins);
    log::Debug() << "isPeakMap\n" << print(isPeak, pads, timebins);
    log::Debug() << "peakCountMap\n" << print(peakCount, pads, timebins);

    log::Debug() << "isPeak:";
    std::vector<unsigned char> isPeakPred(digits.size());
    gpucpy<unsigned char>(state.isPeak, isPeakPred, isPeakPred.size(), queue, true);
    for (size_t i = 0; i < digits.size(); i++)
    {
        log::Debug() << digits[i] << " " << int(isPeakPred[i]);
    }

    resetMaps.call(state, queue);

    queue.finish();

    return true;
}

// vim: set ts=4 sw=4 sts=4 expandtab:
