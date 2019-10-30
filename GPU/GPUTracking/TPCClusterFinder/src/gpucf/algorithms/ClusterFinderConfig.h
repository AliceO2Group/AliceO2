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

#include <gpucf/algorithms/ChargemapLayout.h>
#include <gpucf/algorithms/ClusterBuilder.h>

#include <cstddef>
#include <iosfwd>

namespace gpucf
{

struct ClusterFinderConfig {
#define CLUSTER_FINDER_FLAG(name, val, def, desc) bool name = val;
#include <gpucf/algorithms/ClusterFinderFlags.def>

#define CLUSTER_FINDER_PARAM(name, val, def, desc) int name = val;
#include <gpucf/algorithms/ClusterFinderFlags.def>

  ChargemapLayout layout = ChargemapLayout::TimeMajor;

  ClusterBuilder clusterbuilder = ClusterBuilder::Naive;
};

std::ostream& operator<<(std::ostream&, const ClusterFinderConfig&);

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
