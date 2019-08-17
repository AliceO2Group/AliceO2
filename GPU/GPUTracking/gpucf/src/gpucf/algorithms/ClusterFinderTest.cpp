#include "ClusterFinderTest.h"

#include <gpucf/common/ClusterMap.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>
#include <gpucf/debug/DigitDrawer.h>

#include <shared/constants.h>


using namespace gpucf;


ClusterFinderTest::ClusterFinderTest(
        ClusterFinderConfig cfg, 
        size_t digitnum, 
        ClEnv env)
    : ClusterFinder(cfg, digitnum, env)
    , gt(cfg)
{
}

void ClusterFinderTest::run(nonstd::span<const Digit> digits)
{
    this->digits = digits;

    res = gt.run(digits);

    log::Debug() << "send Digits to GPU";
    digitsToGPU.call(state, digits, queue);

    ASSERT(state.digitnum == digits.size())
        << "state.digitnum = " << state.digitnum;

    log::Debug() << "Fill charge map";
    fillChargeMap.call(state, queue);

    log::Debug() << "Look for peaks";
    findPeaks.call(state, queue);

    checkIsPeaks(res.isPeak);

    compactPeaks.call(state, queue);

    ASSERT(state.peaknum <= state.digitnum);

    checkPeaks(res.peaks);

    countPeaks.call(state, queue);

    computeCluster.call(state, queue);

    checkCluster(res.peaks, res.cluster);


    queue.finish();
}

void ClusterFinderTest::checkIsPeaks(nonstd::span<const unsigned char> isPeakGT)
{
    std::vector<unsigned char> isPeak(state.digitnum);

    gpucpy<unsigned char>(state.isPeak, isPeak, isPeak.size(), queue, true);

    size_t correctPeaks = 0;
    std::vector<size_t> wrongIds;
    for (size_t i = 0; i < std::min(isPeak.size(), isPeakGT.size()); i++)
    {
        ASSERT(isPeak[i] == 0 || isPeak[i] == 1);
        bool ok = (isPeak[i] == isPeakGT[i]);
        correctPeaks += ok;

        if (!ok)
        {
            wrongIds.push_back(i);
        }
    }

    float correctFraction = 
        std::min(correctPeaks, isPeakGT.size())
            / float(std::max(correctPeaks, isPeakGT.size()));

    bool isPeaksOk = correctFraction >= 0.99f;

    if (!isPeaksOk)
    {
        DigitDrawer drawer(digits, res.isPeak, isPeak);

        for (size_t i = 0; i < std::min(size_t(10), wrongIds.size()); i++)
        {
            log::Info() << wrongIds[i] << "\n" << drawer.drawArea(digits[wrongIds[i]], 3);
        }
    }

    ASSERT(isPeaksOk) << "\n  correctFraction = " << correctFraction;

    log::Success() << "isPeaks: OK (" << correctFraction << " correct)";

}

void ClusterFinderTest::checkPeaks(const std::vector<Digit> &peakGT)
{
    std::vector<Digit> peaks(state.peaknum);

    gpucpy<Digit>(state.peaks, peaks, peaks.size(), queue, true);

    Map<bool> peakLookup(peakGT, true, false);
    log::Info() << "GT has   " << peakGT.size() << " peaks.";
    log::Info() << "Cf found " << peaks.size() << " peaks.";

    size_t correctPeaks = 0;
    std::vector<size_t> wrongIds;
    for (size_t i = 0; i < peaks.size(); i++)
    {
        bool ok = peakLookup[peaks[i]];
        correctPeaks += ok;

        if (!ok)
        {
            wrongIds.push_back(i);
        }
    }

    float correctFraction = float(correctPeaks) / peaks.size();

    const float threshold = (state.cfg.halfs) ? 0.98f : 0.99f;

    bool peaksOk = correctFraction >= threshold;

    if (!peaksOk)
    {
        DigitDrawer drawer(digits, peakGT, peaks);

        for (size_t i = 0; i < std::min(size_t(10), wrongIds.size()); i++)
        {
            log::Info() << wrongIds[i] << "\n" << drawer.drawArea(peaks[wrongIds[i]], 3);
        }
    }

    ASSERT(peaksOk) << "\n" << correctFraction;

    log::Success() << "Peaks: OK (" << correctFraction << " correct)";
}

void ClusterFinderTest::checkCluster(
        const std::vector<Digit> &peaksGT, 
        const std::vector<Cluster> &clusterGT)
{
    log::Info() << "Testing cluster...";

    std::vector<Cluster> cluster = clusterToCPU.call(state, queue);

    log::Info() << "GT has   " << clusterGT.size() << " cluster.";
    log::Info() << "Cf found " << cluster.size() << " cluster.";


    ClusterMap clpos;
    clpos.addAll(clusterGT);
    clpos.setClusterEqParams(0.f, 0.f,
            Cluster::Field_all
              ^ (Cluster::Field_timeSigma | Cluster::Field_padSigma));

    std::vector<Digit> peaks(state.peaknum);
    gpucpy<Digit>(state.peaks, peaks, peaks.size(), queue, true);

    DigitDrawer drawer(digits, peaksGT, peaks);

    size_t correctCluster = 0;
    size_t printCluster = 3;
    for (const Cluster &c : cluster)
    {
        bool posOk = clpos.contains(c);

        if (!posOk && printCluster > 0)
        {
            log::Debug() 
                << "Print around cluster:\n"
                << drawer.drawArea(
                    Digit{0, c.globalRow(), int(c.padMean), int(c.timeMean)},
                    4);
            printCluster--;
        }

        correctCluster += posOk;
    }

    float correctFraction = float(correctCluster) / cluster.size();

    const float threshold = (state.cfg.halfs) ? 0.98f : 0.99f;
    ASSERT(correctFraction >= threshold) << "\n correctFraction = " << correctFraction;

    log::Success() << "Cluster: OK (" << correctFraction << " correct)";

    ClusterMap clall;
    clall.addAll(clusterGT);

    size_t correctClusterAll = 0;
    for (const Cluster &c : cluster)
    {
        /* if (!clall.contains(c)) */
        /* { */
        /*     log::Debug() << c; */
        /* } */
        correctClusterAll += clall.contains(c);
    }

    log::Info() << "Correct cluster: " << correctCluster
                << ", With sigma: " << correctClusterAll;
}


void ClusterFinderTest::printInterimValues(
        const std::vector<unsigned char> &pred,
        size_t wgsize)
{
    log::Debug() << "Interim values: ";

    size_t i = 0;
    size_t val = 0;
    for (unsigned char p : pred)
    {
        val += p; 
        if (i == wgsize-1)
        {
            log::Debug() << "  " << val;
            val = 0;
            i   = 0;
        }
        else
        {
            i++;
        }
    }
}

// vim: set ts=4 sw=4 sts=4 expandtab:
