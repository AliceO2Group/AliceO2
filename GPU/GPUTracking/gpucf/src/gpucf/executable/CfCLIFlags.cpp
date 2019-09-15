// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "CfCLIFlags.h"


using namespace gpucf;


CfCLIFlags::CfCLIFlags(args::Group &/*required*/, args::Group &optional)
    : cfconfig(optional, "Clusterfinder config")
    #define CLUSTER_FINDER_FLAG(name, val, def, desc) \
        , name(cfconfig, "", desc, {#name})
    #include <gpucf/algorithms/ClusterFinderFlags.def>
    #define CLUSTER_FINDER_PARAM(name, val, def, desc) \
        , name(cfconfig, "X", desc, {#name})
    #include <gpucf/algorithms/ClusterFinderFlags.def>
    #define MEMORY_LAYOUT(name, def, desc) \
        , layout##name(cfconfig, "", desc, {"layout" #name})
    #include <gpucf/algorithms/ClusterFinderFlags.def>
    #define CLUSTER_BUILDER(name, def, desc) \
        , builder##name(cfconfig, "", desc, {"builder" #name})
    #include <gpucf/algorithms/ClusterFinderFlags.def>
{
}

ClusterFinderConfig CfCLIFlags::asConfig()
{
    ClusterFinderConfig config;

    #define CLUSTER_FINDER_FLAG(name, val, def, desc) \
        config.name = name;
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define CLUSTER_FINDER_PARAM(name, val, def, desc) \
        if (name) \
        { \
            config.name = name.Get(); \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define MEMORY_LAYOUT(name, def, desc) \
        if (layout##name) \
        { \
            config.layout = ChargemapLayout::name; \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    #define CLUSTER_BUILDER(name, def, desc) \
        if (builder##name) \
        { \
            config.clusterbuilder = ClusterBuilder::name; \
        }
    #include <gpucf/algorithms/ClusterFinderFlags.def>

    return config;
}

// vim: set ts=4 sw=4 sts=4 expandtab:

