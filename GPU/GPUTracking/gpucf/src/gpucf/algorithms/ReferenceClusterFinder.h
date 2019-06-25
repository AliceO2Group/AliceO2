#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>

#include <vector>


namespace gpucf
{

class ReferenceClusterFinder
{

public:
    struct Result
    {
        std::vector<Cluster> cluster;
        std::vector<Digit>  peaks;    
    };

    ReferenceClusterFinder(ClusterFinderConfig);

    Result run(nonstd::span<const Digit>);

private:

    using PeakCount = unsigned char;

    enum PCMask : PeakCount
    {
        PCMASK_HAS_3X3_PEAKS = 0x80,
        PCMASK_PEAK_COUNT    = 0x7F,
    };

    static_assert((PCMASK_HAS_3X3_PEAKS ^ PCMASK_PEAK_COUNT) == 0xFF);


    bool isPeak(const Digit &, const Map<float> &);

    Map<PeakCount> makePeakCountMap(
            nonstd::span<const Digit>, 
            nonstd::span<const Digit>, 
            bool);

    PeakCount countPeaks(const Digit &, const Map<bool> &);

    Cluster clusterize(
            const Digit &, 
            const Map<float> &, 
            const Map<PeakCount> &);

    void update(Cluster &, float, int, int);
    void finalize(Cluster &, const Digit &);

    ClusterFinderConfig config;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
