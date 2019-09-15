// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>
#include <gpucf/common/Cluster.h>
#include <gpucf/common/Digit.h>
#include <gpucf/common/Map.h>
#include <gpucf/common/View.h>
#include <gpucf/noisesuppression/Delta.h>
#include <gpucf/noisesuppression/NoiseSuppressionOverArea.h>

#include <vector>


namespace gpucf
{

class ReferenceClusterFinder
{

public:
    struct Result
    {
        std::vector<Cluster> cluster;
        std::vector<Digit>   peaks;    
        std::vector<Digit>   filteredPeaks;    
        std::vector<unsigned char> isPeak;
    };

    ReferenceClusterFinder(ClusterFinderConfig);

    SectorMap<Result> runOnSectors(const SectorMap<std::vector<Digit>> &);
    Result run(View<Digit>);

private:

    using PeakCount = unsigned char;


    enum PCMask : PeakCount
    {
        PCMASK_HAS_3X3_PEAKS = 0x80,
        PCMASK_PEAK_COUNT    = 0x7F,
    };

    static_assert((PCMASK_HAS_3X3_PEAKS ^ PCMASK_PEAK_COUNT) == 0xFF, "");

    static const std::unordered_map<Delta, std::vector<Delta>> innerToOuter;
    static const std::unordered_map<Delta, std::vector<Delta>> innerToOuterInv;

    Map<PeakCount> makePeakCountMap(
            nonstd::span<const Digit>, 
            nonstd::span<const Digit>, 
            const Map<float> &,
            bool);

    PeakCount countPeaks(const Digit &, const Map<bool> &, const Map<float> &);

    Cluster clusterize(
            const Digit &, 
            const Map<float> &, 
            const Map<PeakCount> &);

    void update(Cluster &, float, int, int);
    void finalize(Cluster &, const Digit &);

    ClusterFinderConfig config;

    NoiseSuppressionOverArea noiseSuppression;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
