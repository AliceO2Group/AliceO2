#pragma once

#include <gpucf/common/Digit.h>
#include <gpucf/common/Cluster.h>

#include <vector>


class ClEnv;

class GPUClusterFinder
{
public:
    void run(ClEnv &, const std::vector<Digit> &);

private:
    size_t getNumOfRows(const std::vector<Digit> &);
    void printClusters(const std::vector<int> &, const std::vector<Cluster> &);

};

// vim: set ts=4 sw=4 sts=4 expandtab:
