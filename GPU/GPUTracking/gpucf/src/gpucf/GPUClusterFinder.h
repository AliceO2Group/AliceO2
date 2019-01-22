#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Cluster.h>

#include <vector>


namespace gpucf
{

class ClEnv;

class GPUClusterFinder
{
public:
    struct Result
    {
        std::vector<Cluster> clusters;
        std::vector<Digit>  peaks;
    };

    Result run(ClEnv &, const std::vector<Digit> &);

private:
    static void printClusters(
            const std::vector<int> &, 
            const std::vector<Cluster> &,
            size_t);

    static std::vector<Cluster> filterCluster(
            const std::vector<int> &,
            const std::vector<Cluster> &);

    static std::vector<Digit> findPeaks(
            const std::vector<int> &,
            const std::vector<Digit> &);

};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
