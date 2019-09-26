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
#include <gpucf/common/ClusterMap.h>
#include <gpucf/common/float.h>

#include <nonstd/optional.h>
#include <nonstd/span.h>

#include <unordered_map>
#include <vector>

namespace gpucf
{

class ClusterChecker
{

 public:
  ClusterChecker(nonstd::span<const Cluster>);

  bool verify(nonstd::span<const Cluster>, bool showExamples = true);

 private:
  using ClusterPair = std::pair<Cluster, Cluster>;

  std::vector<Cluster> findWrongClusters(nonstd::span<const Cluster>);

  std::vector<ClusterPair> findTruth(nonstd::span<const Cluster>);

  void findAndLogTruth(
    nonstd::span<const Cluster>,
    const std::string& testPrefix,
    bool showExample,
    float,
    float,
    Cluster::FieldMask);

  void printDuplicates(
    nonstd::span<const Cluster>,
    Cluster::FieldMask);

  ClusterMap truth;
};

} // namespace gpucf

// vim: set ts=4 sw=4 sts=4 expandtab:
