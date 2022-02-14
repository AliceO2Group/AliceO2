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
#include <unordered_set>
#include <vector>
#include <gsl/gsl>
#include "DataFormatsMID/Cluster.h"
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
  /// Initializes the clusterizer
  /// \param func Function to associate the cluster index with the corresponding pre-clusters
  bool init(std::function<void(size_t, size_t)> func = [](size_t, size_t) {});

  /// Builds the clusters from the pre-clusters in one event
  /// \param preClusters gsl::span of PreClusters objects in the same event
  /// \param accumulate Flag to decide if one needs to reset the output clusters at each event
  void process(gsl::span<const PreCluster> preClusters, bool accumulate = false);

  /// Builds the clusters from the pre-clusters in the timeframe
  /// \param preClusters gsl::span of PreClusters objects in the timeframe
  /// \param rofRecords RO frame records
  void process(gsl::span<const PreCluster> preClusters, gsl::span<const ROFRecord> rofRecords);

  /// Gets the vector of reconstructed clusters
  const std::vector<Cluster>& getClusters() { return mClusters; }

  /// Gets the vector of clusters RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  /// Resets the clusters
  void reset();

  /// Fills the structure with pre-clusters
  /// \param preClusters gsl::span of PreClusters
  /// \return true if preClusters is not empty
  bool loadPreClusters(gsl::span<const PreCluster>& preClusters);

  /// Makes the clusters and stores them
  /// \param pcs PreClusters for one Detection Element
  /// \return true
  bool makeClusters(PreClustersDE& pcs);

  /// Makes the cluster from pre-clusters: simple case
  /// \param areaBP Pre-cluster in the Bending Plane
  /// \param areaNBP Pre-cluster in the Non-Bending Plane
  /// \param deId Detection element ID
  void makeCluster(const MpArea& areaBP, const MpArea& areaNBP, uint8_t deId);

  /// Makes the cluster from mono-cathodic pre-cluster
  /// \param area Pre-cluster in the Bending or Non-Bending Plane
  /// \param deId Detection element ID
  /// \param isBP Fired cathode
  void makeCluster(const MpArea& area, const uint8_t deId, int cathode);

  /// Makes the cluster from pre-clusters: general case
  /// \param pcBP Pre-cluster in the Bending Plane
  /// \param pcBPNeigh Neighbour pre-cluster in the Bending Plane
  /// \param pcNBP Pre-cluster in the Non-Bending Plane
  /// \param deId Detection element ID
  void makeCluster(const PreClustersDE::BP& pcBP, const PreClustersDE::BP& pcBPNeigh, const PreClustersDE::NBP& pcNBP, uint8_t deId);

  gsl::span<const PreCluster> mPreClusters;                    //!< Input pre-clusters
  std::unordered_map<uint8_t, PreClustersDE> mPreClustersDE{}; //!< Pre-clusters per Detection Element
  std::unordered_set<uint8_t> mActiveDEs{};                    //!< List of active detection elements for event
  PreClusterHelper mPreClusterHelper{};                        //!< Helper for pre-clusters
  std::vector<Cluster> mClusters{};                            ///< List of clusters
  std::vector<ROFRecord> mROFRecords{};                        ///< List of cluster RO frame records
  size_t mPreClusterOffset{0};                                 //!< RO offset for pre-cluster

  std::function<void(size_t, size_t)> mFunction{[](size_t, size_t) {}}; ///! Function to keep track of input-output relation
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERIZER_H */
