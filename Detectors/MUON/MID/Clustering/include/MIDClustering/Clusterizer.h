// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDClustering/Clusterizer.h
/// \brief  Cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   24 October 2016

#ifndef O2_MID_CLUSTERIZER_H
#define O2_MID_CLUSTERIZER_H

#include <functional>
#include <unordered_map>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDBase/MpArea.h"
#include "MIDClustering/PreCluster.h"
#include "MIDClustering/PreClusterHelper.h"
#include "MIDClustering/PreClustersDE.h"

namespace o2
{
namespace mid
{
/// Clusterizing algorithm for MID
class Clusterizer
{
 public:
  bool init(std::function<void(size_t, size_t)> func = [](size_t, size_t) {});
  void process(gsl::span<const PreCluster> preClusters, bool accumulate = false);
  void process(gsl::span<const PreCluster> preClusters, gsl::span<const ROFRecord> rofRecords);

  /// Gets the vector of reconstructed clusters
  const std::vector<Cluster2D>& getClusters() { return mClusters; }

  /// Gets the vector of clusters RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  void reset();
  bool loadPreClusters(gsl::span<const PreCluster>& preClusters);

  bool makeClusters(PreClustersDE& pcs);
  void makeCluster(const MpArea& areaBP, const MpArea& areaNBP, const int& icolumn, const int& deIndex);
  void makeCluster(const PreClustersDE::BP& pcBP, const PreClustersDE::BP& pcBPNeigh, const PreClustersDE::NBP& pcNBP, const int& deIndex);

  const gsl::span<const PreCluster>* mPreClusters{nullptr}; ///! Input pre-clusters
  std::unordered_map<int, PreClustersDE> mPreClustersDE{};  ///! Sorted pre-clusters
  std::unordered_map<int, bool> mActiveDEs{};               ///! List of active detection elements for event
  PreClusterHelper mPreClusterHelper{};                     ///! Helper for pre-clusters
  std::vector<Cluster2D> mClusters{};                       ///< List of clusters
  std::vector<ROFRecord> mROFRecords{};                     ///< List of cluster RO frame records
  size_t mPreClusterOffset{0};                              //!< RO offset for pre-cluster
  std::function<void(size_t, size_t)> mFunction;            ///! Function to keep track of input-output relation
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERIZER_H */
