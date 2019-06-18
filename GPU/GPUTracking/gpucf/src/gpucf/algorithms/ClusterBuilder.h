#pragma once

namespace gpucf
{

enum class ClusterBuilder
{
    #define CLUSTER_BUILDER(name, def, desc) name,    
    #include <gpucf/algorithms/ClusterFinderFlags.def>
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
