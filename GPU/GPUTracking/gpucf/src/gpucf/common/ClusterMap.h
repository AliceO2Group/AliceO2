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

#include <gpucf/common/Cluster.h>
#include <gpucf/common/float.h>

#include <nonstd/span.h>

#include <vector>

namespace gpucf
{

class ClusterMap
{

 public:
  void add(const Cluster&);
  void addAll(nonstd::span<const Cluster>);

  bool contains(const Cluster&) const;
  nonstd::optional<Cluster> tryLookup(const Cluster&) const;

  std::vector<Cluster> findDuplicates() const;

  void setClusterEqParams(float, float, Cluster::FieldMask);

  size_t size() const;

  Cluster findClosest(const Cluster&) const;

 private:
  std::unordered_map<int, std::vector<Cluster>> clusters;

  float epsilonSmall = FEQ_EPSILON_SMALL;
  float epsilonBig = FEQ_EPSILON_BIG;
  Cluster::FieldMask mask = Cluster::Field_all;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
