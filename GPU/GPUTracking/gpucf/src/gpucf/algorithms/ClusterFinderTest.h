#pragma once

#include <gpucf/algorithms/ClusterFinder.h>
#include <gpucf/algorithms/ReferenceClusterFinder.h>


namespace gpucf
{

class ClusterFinderTest : public ClusterFinder 
{

public:
    ClusterFinderTest(ClusterFinderConfig, size_t, ClEnv);

    void run(nonstd::span<const Digit>);

private:
    ReferenceClusterFinder gt;
    ReferenceClusterFinder::Result res;
    nonstd::span<const Digit> digits;

    void checkIsPeaks(nonstd::span<const unsigned char>);

    void checkPeaks(const std::vector<Digit> &);

    void checkCluster(
            const std::vector<Digit> &, 
            const std::vector<Cluster> &);

    void printInterimValues(const std::vector<unsigned char> &, size_t);
    
};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
