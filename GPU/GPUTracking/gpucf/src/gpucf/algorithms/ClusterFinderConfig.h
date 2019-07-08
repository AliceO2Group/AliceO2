#pragma once

#include <gpucf/algorithms/ChargemapLayout.h>
#include <gpucf/algorithms/ClusterBuilder.h>

#include <cstddef>
#include <iosfwd>


namespace gpucf
{

struct ClusterFinderConfig
{
    #define CLUSTER_FINDER_FLAG(name, val, def, desc) bool name = val;
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    ChargemapLayout layout = ChargemapLayout::TimeMajor;

    ClusterBuilder clusterbuilder = ClusterBuilder::Naive;
};

std::ostream &operator<<(std::ostream &, const ClusterFinderConfig &);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
