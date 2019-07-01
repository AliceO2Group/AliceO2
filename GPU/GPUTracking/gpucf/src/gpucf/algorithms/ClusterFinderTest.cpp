#include "ClusterFinderTest.h"

#include <gpucf/common/ClusterMap.h>
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

    checkCompactedCluster(res.peaks, res.cluster);

    queue.finish();
}

void ClusterFinderTest::checkIsPeaks(const std::vector<bool> &isPeakGT)
{
    std::vector<unsigned char> isPeak(state.digitnum);

    gpucpy<unsigned char>(state.isPeak, isPeak, isPeak.size(), queue, true);

    if (state.cfg.halfs)
    {
        return;
    } 
    else 
    {
        ASSERT(isPeakGT.size() == isPeak.size());

        bool correctPeaks = true;
        for (size_t i = 0; i < isPeak.size(); i++)
        {
            bool ok = (isPeak[i] == isPeakGT[i]);

            /* if (!ok) */
            /* { */
            /*     log::Debug() << "i = " << i << "; peak = " << int(isPeak[i]); */
            /* } */

            correctPeaks &= ok;
        }
        ASSERT(correctPeaks);

        log::Success() << "isPeaks: OK";
    }

}

void ClusterFinderTest::checkPeaks(const std::vector<Digit> &peakGT)
{
    std::vector<Digit> peaks(state.peaknum);

    gpucpy<Digit>(state.peaks, peaks, peaks.size(), queue, true);

    if (state.cfg.halfs)
    {
        Map<bool> peakLookup(peaks, true, false);
        log::Info() << "GT has   " << peakGT.size() << " peaks.";
        log::Info() << "Cf found " << peaks.size() << " peaks.";

        size_t correctPeaks = 0;
        for (const Digit &d : peakGT)
        {
            correctPeaks += peakLookup[d];
        }

        log::Info() << "Correct peaks: " << correctPeaks;
    }
    else
    {
        ASSERT(peakGT.size() == peaks.size());

        bool correctPeaks = true;
        for (size_t i = 0; i < peaks.size(); i++)
        {
            bool ok = (peaks[i] == peakGT[i]);

            if (!ok)
            {
                log::Debug() << "i = " << i << "; peak = " << peaks[i];
            }

            correctPeaks &= ok;
        }
        ASSERT(correctPeaks);
    
        log::Success() << "peaks: OK";
    }
}

void ClusterFinderTest::checkCluster(
        const std::vector<Digit> &peaks, 
        const std::vector<Cluster> &clusterGT)
{
    std::vector<ClusterNative> cn(state.peaknum);

    gpucpy<ClusterNative>(state.clusterNative, cn, cn.size(), queue, true);

    std::vector<Cluster> cluster;
    for (size_t i = 0; i < cn.size(); i++)
    {
        cluster.emplace_back(peaks[i].cru(), peaks[i].localRow(), cn[i]);
    }

    if (state.cfg.halfs)
    {
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
    else
    {
        ASSERT(clusterGT.size() == cn.size());
        ASSERT(peaks.size() == cn.size());


        bool clusterOk = true;
        int clusterPrinted = 10;
        for (size_t i = 0; i < cluster.size(); i++)
        {
            bool ok = (cluster[i].eq(clusterGT[i], 0.f, 0.f, Cluster::Field_all));
            clusterOk &= ok;

            if (!ok && clusterPrinted > 0)
            {
                log::Debug() << "i  = " << i
                    << "\n c  = " << cluster[i]
                    << "\n gt = " << clusterGT[i];

                clusterPrinted--;
            }
        }

        ASSERT(clusterOk);

        log::Success() << "cluster: OK";
    }

}

void ClusterFinderTest::checkCompactedCluster(
        const std::vector<Digit> &peaks,
        const std::vector<Cluster> &clusterGT)
{
    if (state.cfg.halfs)
    {
        return;
    }

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
