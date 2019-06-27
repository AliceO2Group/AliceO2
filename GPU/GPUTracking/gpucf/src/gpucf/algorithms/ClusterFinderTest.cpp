#include "ClusterFinderTest.h"

#include <gpucf/common/log.h>

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

    digitsToGPU.call(state, digits, queue);

    ASSERT(state.digitnum == size_t(digits.size()))
        << "state.digitnum = " << state.digitnum;

    fillChargeMap.call(state, queue);
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

    checkCompactedCluster(res.peaks, res.cluster);

    /* nativeToRegular.call(state, queue); */

    queue.finish();
}

void ClusterFinderTest::checkIsPeaks(const std::vector<bool> &isPeakGT)
{
    std::vector<unsigned char> isPeak(state.digitnum);

    gpucpy<unsigned char>(state.isPeak, isPeak, isPeak.size(), queue, true);

    ASSERT(isPeakGT.size() == isPeak.size());

    for (size_t i = 0; i < isPeak.size(); i++)
    {
        ASSERT(isPeakGT[i] == isPeak[i]);
    }

    log::Success() << "isPeaks: OK";
}

void ClusterFinderTest::checkPeaks(const std::vector<Digit> &peakGT)
{
    std::vector<Digit> peaks(state.peaknum);

    gpucpy<Digit>(state.peaks, peaks, peaks.size(), queue, true);

    ASSERT(peakGT.size() == peaks.size());

    for (size_t i = 0; i < peaks.size(); i++)
    {
        ASSERT(peaks[i] == peakGT[i]);
    }
    
    log::Success() << "peaks: OK";
}

void ClusterFinderTest::checkCluster(
        const std::vector<Digit> &peaks, 
        const std::vector<Cluster> &clusterGT)
{
    std::vector<ClusterNative> cn(state.peaknum);

    ASSERT(clusterGT.size() == cn.size());
    ASSERT(peaks.size() == cn.size());

    gpucpy<ClusterNative>(state.clusterNative, cn, cn.size(), queue, true);

    std::vector<Cluster> cluster;
    for (size_t i = 0; i < cn.size(); i++)
    {
        cluster.emplace_back(peaks[i].cru(), peaks[i].localRow(), cn[i]);
    }

    for (size_t i = 0; i < cluster.size(); i++)
    {
        ASSERT(cluster[i].eq(clusterGT[i], 0.f, 0.f, Cluster::Field_all));
    }

    log::Success() << "cluster: OK";
}

void ClusterFinderTest::checkCompactedCluster(
        const std::vector<Digit> &peaks,
        const std::vector<Cluster> &clusterGT)
{
    std::vector<ClusterNative> cn(state.cutoffClusternum);
    std::vector<unsigned char> cutoff(state.peaknum);

    ASSERT(cn.size() <= clusterGT.size());

    gpucpy<ClusterNative>(
            state.clusterNativeCutoff, 
            cn, 
            cn.size(), 
            queue, 
            true);

    gpucpy<cl_uchar>(
            state.aboveQTotCutoff, 
            cutoff, 
            cutoff.size(), 
            queue, 
            true);

    size_t cutpos = 0;
    for (size_t i = 0; i < clusterGT.size(); i++)
    {
        ASSERT(cutoff[i] == 0 || cutoff[i] == 1);
        if (cutoff[i])
        {
            ASSERT(clusterGT[i].Q >= QTOT_THRESHOLD || !state.cfg.qtotCutoff);
            ASSERT(Cluster(peaks[i], cn[cutpos]).eq(
                        clusterGT[i], 0.f, 0.f, Cluster::Field_all));
            cutpos++;
        }
        else
        {
            ASSERT(clusterGT[i].Q < QTOT_THRESHOLD);
        }
    }

    if (cutpos != state.cutoffClusternum) 
    {
        printInterimValues(cutoff, 256);

        size_t cutoffGT = 0;
        for (const Cluster &c : clusterGT)
        {
            cutoffGT += (c.Q >= QTOT_THRESHOLD || !state.cfg.qtotCutoff);
        }

        DBG(cutoffGT);

        ASSERT(cutpos == state.cutoffClusternum) 
            << "\n  cutpos = " << cutpos
            << "\n  state.cutoffClusternum = " << state.cutoffClusternum
            << "\n  clusternum = " << peaks.size();
    }


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
