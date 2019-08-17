#include "ReferenceClusterFinder.h"

#include <gpucf/algorithms/cpu.h>
#include <gpucf/common/RowInfo.h>

#include <shared/constants.h>

#include <cmath>


using namespace gpucf;


const std::unordered_map<Delta, std::vector<Delta>> ReferenceClusterFinder::innerToOuter =
    {
        { {-1, -1}, { {-2, -1}, {-2, -2}, {-1, -2} } },
        { {-1, 0}, { {-2, 0} } },
        { {-1, 1},{ {-2, 1}, {-2, 2}, {-1, 2} } },
        { {0, -1},{ {0, -2} } },
        { {0, 1},{ {0, 2} } },
        { {1, -1},{ {2, -1}, {2, -2}, {1, -2} } },
        { {1, 0}, { {2, 0} } },
        { {1, 1},{ {2, 1}, {2, 2}, {1, 2} } },
    };

const std::unordered_map<Delta, std::vector<Delta>> ReferenceClusterFinder::innerToOuterInv =
    {
        { {-1, -1}, { {-2, -2} } },
        { {-1, 0}, { {-2, -1}, {-2, 0}, {-2, 1} } },
        { {-1, 1},{ {-2, 2} } },
        { {0, -1},{ {-1, -2}, {0, -2}, {1, -2} } },
        { {0, 1},{ {-1, 2}, {0, 2}, {1, 2} } },
        { {1, -1},{ {2, -2}  }},
        { {1, 0}, { {2, -1}, {2, 0}, {2, 1} } },
        { {1, 1},{ {2, 2} } },
    };



ReferenceClusterFinder::ReferenceClusterFinder(ClusterFinderConfig config)
    : config(config)
{
}

SectorMap<ReferenceClusterFinder::Result> ReferenceClusterFinder::runOnSectors(
        const SectorMap<std::vector<Digit>> &digits)
{
    SectorMap<Result> results;

    for (size_t i = 0; i < TPC_SECTORS; i++)
    {
        results[i] = run(digits[i]);
    }

    return results;
}

ReferenceClusterFinder::Result ReferenceClusterFinder::run(View<Digit> digits)
{
    Map<float> chargemap(digits, [](const Digit &d) { return d.charge; }, 0.f);

    std::vector<Digit> peaks;
    std::vector<unsigned char> isPeakPred;

    DBG(config.qmaxCutoff);

    for (const Digit &d : digits)
    {
        bool p = isPeak(d, chargemap, config.qmaxCutoff);
        isPeakPred.push_back(p);

        if (p)
        {
            peaks.push_back(d);
        }
    }

    Map<PeakCount> peakcount = makePeakCountMap(digits, peaks, chargemap, true);

    std::vector<Cluster> clusters;

    std::transform(
            peaks.begin(), 
            peaks.end(),
            std::back_inserter(clusters),
            [=](const Digit &d) { 
            return clusterize(d, chargemap, peakcount); });

    return {clusters, peaks, isPeakPred};
}


Cluster ReferenceClusterFinder::clusterize(
        const Digit &d, 
        const Map<float> &chargemap,
        const Map<PeakCount> &peakcount)
{
    Cluster c;

    for (const auto &p : innerToOuter)
    {
        const Delta &inner = p.first; 

        PeakCount pc = PCMASK_PEAK_COUNT & peakcount[{d, inner.pad, inner.time}];
        float q = chargemap[{d, inner.pad, inner.time}] / pc;
        update(c, q, inner.pad, inner.time);

        if (q <= CHARGE_THRESHOLD)
        {
            continue;
        }

        for (const auto &outer : p.second)
        {
            PeakCount pc = peakcount[{d, outer.pad, outer.time}];
            if (PCMASK_HAS_3X3_PEAKS & pc)
            {
                continue;
            }

            pc = PCMASK_PEAK_COUNT & pc;
            float q = chargemap[{d, outer.pad, outer.time}] / pc;
            update(c, q, outer.pad, outer.time); 
        }
    }

    finalize(c, d);

    return c;
}


Map<ReferenceClusterFinder::PeakCount> ReferenceClusterFinder::makePeakCountMap(
        nonstd::span<const Digit> digits,
        nonstd::span<const Digit> peaks,
        const Map<float> &chargemap,
        bool deconv)
{
    if (!deconv)
    {
        return {{}, [](const Digit &) { return 1; }, 1};
    }

    Map<bool> peakmap(peaks, true, false);
    return {
        digits, 
            [=](const Digit &d) { return countPeaks(d, peakmap, chargemap); },
            1};
}

ReferenceClusterFinder::PeakCount ReferenceClusterFinder::countPeaks(
        const Digit &d, 
        const Map<bool> &peakmap,
        const Map<float> &chargemap)
{
    if (peakmap[d])
    {
        return PCMASK_HAS_3X3_PEAKS | 1;
    }

    PeakCount innerPeaks = 0;
    PeakCount outerPeaks = 0;

    for (const auto &p : innerToOuterInv)
    {
        Delta inner = p.first;

        innerPeaks += peakmap[{d, inner.pad, inner.time}];

        if (chargemap[{d, inner.pad, inner.time}] <= CHARGE_THRESHOLD)
        {
            continue;
        }

        for (const auto &outer : p.second)
        {
            outerPeaks += peakmap[{d, outer.pad, outer.time}];
        }
    }

    return (innerPeaks > 0) ? (PCMASK_HAS_3X3_PEAKS | innerPeaks) : outerPeaks;

}


void ReferenceClusterFinder::update(Cluster &c, float q, int dp, int dt)
{
    c.Q         += q;
    c.padMean   += q*dp;
    c.timeMean  += q*dt;
    c.padSigma  += q*dp*dp;
    c.timeSigma += q*dt*dt;
}

void ReferenceClusterFinder::finalize(Cluster &c, const Digit &peak)
{
    float qtot      = c.Q + peak.charge;
    float padMean   = c.padMean;
    float timeMean  = c.timeMean;
    float padSigma  = c.padSigma;
    float timeSigma = c.timeSigma;

    padMean   /= qtot;
    timeMean  /= qtot;
    padSigma  /= qtot;
    timeSigma /= qtot;

    padSigma  = std::sqrt(padSigma  - padMean*padMean);
    timeSigma = std::sqrt(timeSigma - timeMean*timeMean);

    padMean  += peak.pad;
    timeMean += peak.time;

    c.Q    = int(qtot);
    c.QMax = int(peak.charge);
    c.padMean = padMean;
    c.timeMean = timeMean;
    c.timeSigma = timeSigma;
    c.padSigma = padSigma;

    c.cru = RowInfo::instance().globalRowToCru(peak.row);
    c.row = RowInfo::instance().globalToLocal(peak.row);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
