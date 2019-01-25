#pragma once

#include <gpucf/common/Cluster.h>


namespace gpucf
{

class ClusterChecker
{

public:
    bool verify(
            const std::vector<Cluster> &, 
            const std::vector<Cluster> &truth);

private:
    size_t countCorrectClusters(
            const std::vector<Cluster> &,
            const std::vector<Cluster> &);
    
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
