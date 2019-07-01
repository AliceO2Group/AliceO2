#pragma once

#include <gpucf/algorithms/ClusterFinder.h>

#include <vector>


namespace gpucf
{

class ClusterFinderProfiler : public ClusterFinder
{

public:
    using ClusterFinder::ClusterFinder;

    std::vector<Step> run(nonstd::span<const Digit>);
    
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
