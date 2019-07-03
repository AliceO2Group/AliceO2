#include "ClusterFinderTest.h"

#include <gpucf/common/ClusterMap.h>
#include <gpucf/common/log.h>
#include <gpucf/common/RowInfo.h>

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
    auto res = gt.run(digits);

    log::Debug() << "send Digits to GPU";
    digitsToGPU.call(state, digits, queue);

    ASSERT(state.digitnum == size_t(digits.size()))
        << "state.digitnum = " << state.digitnum;

    log::Debug() << "Fill charge map";
    fillChargeMap.call(state, queue);

    log::Debug() << "Look for peaks";
    findPeaks.call(state, queue);

    checkIsPeaks(res.isPeak);

    compactPeaks.call(state, queue);

    ASSERT(state.peaknum <= state.digitnum);

    checkPeaks(res.peaks);

    if (state.cfg.splitCharges)
    {
        countPeaks.call(state, queue);
    }

    computeCluster.call(state, queue);

    checkCluster(res.peaks, res.cluster);

    compactCluster.call(state, queue);

    checkCompactedCluster();

    queue.finish();
}

void ClusterFinderTest::checkIsPeaks(const std::vector<bool> &isPeakGT)
{
    std::vector<unsigned char> isPeak(state.digitnum);

    gpucpy<unsigned char>(state.isPeak, isPeak, isPeak.size(), queue, true);

    size_t correctPeaks = 0;
    for (size_t i = 0; i < std::min(isPeak.size(), isPeakGT.size()); i++)
    {
        correctPeaks += (isPeak[i] == isPeakGT[i]);
    }

    float correctFraction = 
        std::min(correctPeaks, isPeakGT.size()) 
            / float(std::max(correctPeaks, isPeakGT.size()));

    ASSERT(correctFraction >= 0.99f);

    log::Success() << "isPeaks: OK (" << correctFraction << " correct)";

}

void ClusterFinderTest::checkPeaks(const std::vector<Digit> &peakGT)
{
    std::vector<Digit> peaks(state.peaknum);

    gpucpy<Digit>(state.peaks, peaks, peaks.size(), queue, true);

    Map<bool> peakLookup(peaks, true, false);
    log::Info() << "GT has   " << peakGT.size() << " peaks.";
    log::Info() << "Cf found " << peaks.size() << " peaks.";

    size_t correctPeaks = 0;
    for (const Digit &d : peakGT)
    {
        correctPeaks += peakLookup[d];
    }

    float correctFraction = float(correctPeaks) / peakGT.size();

    ASSERT(correctFraction >= 0.99f) << "\n" << correctFraction;

    log::Success() << "Peaks: OK (" << correctFraction << " correct)";
}

void ClusterFinderTest::checkCluster(
        const std::vector<Digit> &/*peaks*/, 
        const std::vector<Cluster> &clusterGT)
{
    std::vector<Cluster> cluster = clusterToCPU.call(state, false, queue);

    ClusterMap cl;
    cl.addAll(clusterGT);
    cl.setClusterEqParams(0.f, 0.f, 
            Cluster::Field_all ^ (Cluster::Field_timeSigma 
                | Cluster::Field_padSigma));

    size_t correctCluster = 0;
    for (const Cluster &c : cluster)
    {
        /* if (!cl.contains(c)) */
        /* { */
        /*     log::Debug() << c; */
        /* } */
        correctCluster += cl.contains(c);
    }

    float correctFraction = float(correctCluster) / cluster.size();

    ASSERT(correctFraction >= 0.99f) << "\n correctFraction = " << correctFraction;

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

void ClusterFinderTest::checkCompactedCluster()
{
    ASSERT(state.cutoffClusternum <= state.peaknum);

    std::vector<Cluster> cluster = clusterToCPU.call(state, false, queue);
    std::vector<Cluster> clusterAboveCutoff = 
        clusterToCPU.call(state, true, queue);

    ClusterMap map;
    map.addAll(cluster);

    size_t correctCluster = 0;
    for (const Cluster &c : clusterAboveCutoff)
    {
        ASSERT(c.Q >= QTOT_THRESHOLD || !state.cfg.qtotCutoff); 

        correctCluster += map.contains(c);
    }

    log::Info() << "Correct cutoff cluster: " 
                << float(correctCluster) / clusterAboveCutoff.size();

    /* if (correctCluster != state.cutoffClusternum) */ 
    /* { */
    /*     printInterimValues(, 256); */

    /*     size_t cutoffGT = 0; */
    /*     for (const Cluster &c : clusterGT) */
    /*     { */
    /*         cutoffGT += (c.Q >= QTOT_THRESHOLD || !state.cfg.qtotCutoff); */
    /*     } */

    /*     DBG(cutoffGT); */

    /*     ASSERT(cutpos == state.cutoffClusternum) */ 
    /*         << "\n  cutpos = " << cutpos */
    /*         << "\n  state.cutoffClusternum = " << state.cutoffClusternum */
    /*         << "\n  clusternum = " << state.peaknum; */
    /* } */

    log::Success() << "Compacted cluster: OK";
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
