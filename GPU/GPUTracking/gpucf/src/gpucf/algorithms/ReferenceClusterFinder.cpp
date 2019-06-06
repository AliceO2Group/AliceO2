#include "ReferenceClusterFinder.h"

#include <gpucf/common/RowInfo.h>

#include <cmath>


using namespace gpucf;


ReferenceClusterFinder::ReferenceClusterFinder(ClusterFinderConfig config)
    : config(config)
{
}

std::vector<Cluster> ReferenceClusterFinder::run(
        nonstd::span<const Digit> digits)
{
    Map<float> chargemap(digits, [](const Digit &d) { return d.charge; }, 0.f);

    std::vector<Digit> peaks;
    std::copy_if(
            digits.begin(),
            digits.end(),
            std::back_inserter(peaks),
            [=](const Digit &d) { return isPeak(d, chargemap); });


    Map<PeakCount> peakcount = makePeakCountMap(digits, peaks, config.splitCharges);

    std::vector<Cluster> clusters;

    if (config.splitCharges)
    {
        std::transform(
                peaks.begin(), 
                peaks.end(),
                std::back_inserter(clusters),
                [=](const Digit &d) { 
                    return clusterize(d, chargemap, peakcount); });
    }
    else
    {
        std::transform(
                peaks.begin(), 
                peaks.end(),
                std::back_inserter(clusters),
                [=](const Digit &d) { 
                    return clusterize(d, chargemap, peakcount); });
    }

    return clusters;
}


Cluster ReferenceClusterFinder::clusterize(
        const Digit &d, 
        const Map<float> &chargemap,
        const Map<PeakCount> &peakcount)
{
    Cluster c;

    for (int dp = -1; dp <= 1; dp++)
    {
        for (int dt = -1; dt <= 1; dt++)
        {
            PeakCount pc = PCMASK_PEAK_COUNT & peakcount[{d, dp, dt}];
            float q = chargemap[{d, dp, dt}] / pc;
            update(c, q, dp, dt);
        }
    }

    for (int dp = -2; dp <= 2; dp++)
    {
        for (int dt = -2; dt <= 2; dt++)
        {
            if (std::abs(dt) < 2 && std::abs(dp) < 2)
            {
                continue;
            }

            PeakCount pc = peakcount[{d, dp, dt}];
            if (PCMASK_HAS_3X3_PEAKS & pc)
            {
                continue;
            }

            pc = PCMASK_PEAK_COUNT & pc;
            float q = chargemap[{d, dp, dt}] / pc;
            update(c, q, dp, dt); 
        }
    }

    finalize(c, d);

    return c;
}


bool ReferenceClusterFinder::isPeak(const Digit &d, const Map<float> &chargemap)
{
    float q = d.charge;

    bool peak = true;
    peak &= chargemap[{d, -1, -1}] <= q;
    peak &= chargemap[{d, -1,  0}] <= q;
    peak &= chargemap[{d, -1,  1}] <= q;
    peak &= chargemap[{d,  0, -1}] <= q;
    peak &= chargemap[{d,  0,  1}] <  q;
    peak &= chargemap[{d,  1, -1}] <  q;
    peak &= chargemap[{d,  1,  0}] <  q;
    peak &= chargemap[{d,  1,  1}] <  q;

    return peak;
}

Map<ReferenceClusterFinder::PeakCount> ReferenceClusterFinder::makePeakCountMap(
        nonstd::span<const Digit> digits,
        nonstd::span<const Digit> peaks,
        bool deconv)
{
    if (!deconv)
    {
        return {{}, [](const Digit &) { return 1; }, 1};
    }

    Map<bool> peakmap(peaks, [](const Digit &) { return true; }, false);
    return {
        digits, 
        [=](const Digit &d) { return countPeaks(d, peakmap); },
        1};
}

ReferenceClusterFinder::PeakCount ReferenceClusterFinder::countPeaks(
        const Digit &d, 
        const Map<bool> &peakmap)
{

    if (peakmap[{d, 0, 0}]) 
    {
        return PCMASK_HAS_3X3_PEAKS | 1;
    }

    PeakCount peaks = 0; 

    for (int dp = -1; dp <= 1; dp++)
    {
        for (int dt = -1; dt <= 1; dt++)
        {

            peaks += peakmap[{d, dp, dt}];
        }
    }

    if (peaks > 0)
    {
        return PCMASK_HAS_3X3_PEAKS | peaks;
    }

    for (int dp = -2; dp <= 2; dp++)
    {
        for (int dt = -2; dt <= 2; dt++)
        {
            if (std::abs(dt) < 2 && std::abs(dp) < 2)
            {
                continue;
            }

            peaks += peakmap[{d, dp, dt}];
        }
    }
    
    return peaks;
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
    float qtot      = c.Q;
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

    c.Q    = qtot;
    c.QMax = std::round(peak.charge);
    c.padMean = padMean;
    c.timeMean = timeMean;
    c.timeSigma = timeSigma;
    c.padSigma = padSigma;

    c.cru = RowInfo::instance().globalRowToCru(peak.row);
    c.row = RowInfo::instance().globalToLocal(peak.row);
}

// vim: set ts=4 sw=4 sts=4 expandtab:
