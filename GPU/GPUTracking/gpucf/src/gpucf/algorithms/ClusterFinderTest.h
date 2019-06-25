#pragma once

#include <gpucf/algorithms/ClusterFinder.h>


namespace gpucf
{

class ClusterFinderTest : public ClusterFinder {

public:
    using ClusterFinder::ClusterFinder;

    void run(nonstd::span<const Digit>);
    
};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
