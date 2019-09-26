// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "ClusterFinderConfig.h"

#include <ostream>

using namespace gpucf;

std::ostream& gpucf::operator<<(std::ostream& o, const ClusterFinderConfig& cfg)
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
