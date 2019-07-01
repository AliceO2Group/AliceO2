#include "ClusterFinderConfig.h"

#include <ostream>


using namespace gpucf;


std::ostream &gpucf::operator<<(std::ostream &o, const ClusterFinderConfig &cfg)
{
    return o << "Config{"
    #define CLUSTER_FINDER_FLAG(name, val, def, desc) \
        << " " #name " = " << cfg.name << ","
    #include <gpucf/algorithms/ClusterFinderFlags.def>
        << " layout = " << static_cast<int>(cfg.layout) << ","
        << " builder = " << static_cast<int>(cfg.clusterbuilder)
        << " }";
}

// vim: set ts=4 sw=4 sts=4 expandtab:
