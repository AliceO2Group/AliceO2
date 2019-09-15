// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
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
