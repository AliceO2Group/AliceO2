#pragma once

#include <gpucf/common/Cluster.h>


namespace gpucf
{

class ClusterChecker
{

public:
    bool verify(const std::vector<Cluster> &);

private:
    static bool hasNaN(const Cluster &);
    static bool hasWeirdEntries(const Cluster &);
    
};

}

// vim: set ts=4 sw=4 sts=4 expandtab:
