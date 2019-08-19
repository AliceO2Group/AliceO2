#pragma once

#include <gpucf/algorithms/ClusterFinderConfig.h>

#include <args/args.hxx>


namespace gpucf
{

class CfCLIFlags
{

public:
    CfCLIFlags(args::Group &, args::Group &);

    ClusterFinderConfig asConfig();

private:
    args::Group cfconfig;
#define CLUSTER_FINDER_FLAG(name, val, def, desc) args::Flag name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define CLUSTER_FINDER_PARAM(name, val, def, desc) args::ValueFlag<int> name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define MEMORY_LAYOUT(name, def, desc) args::Flag layout##name;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define CLUSTER_BUILDER(name, def, desc) args::Flag builder##name;
#include <gpucf/algorithms/ClusterFinderFlags.def>
    
};
    
} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
