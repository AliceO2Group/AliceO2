// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

#include <unordered_map>
#include <vector>
#include "MIDBase/Mapping.h"
#include "DataFormatsMID/Cluster2D.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDClustering/PreClusters.h"

namespace o2
{
namespace mid
{
/// Clusterizing algorithm for MID
class Clusterizer
{
 public:
  Clusterizer();
  virtual ~Clusterizer() = default;

  Clusterizer(const Clusterizer&) = delete;
  Clusterizer& operator=(const Clusterizer&) = delete;
  Clusterizer(Clusterizer&&) = delete;
  Clusterizer& operator=(Clusterizer&&) = delete;

  bool init();
  bool process(std::vector<PreClusters>& preClusters);

  /// Gets the array of reconstructes clusters
  const std::vector<Cluster2D>& getClusters() { return mClusters; }

  /// Gets the number of reconstructed clusters
  unsigned long int getNClusters() { return mNClusters; }

 private:
  void reset();

  PreClusters::PreClusterBP getNeighbour(int icolumn, bool skipPaired);

  Cluster2D& nextCluster();
  bool makeClusters(PreClusters& pcs);
  void makeCluster(PreClusters::PreClusterBP& clBend, PreClusters::PreClusterNBP& clNonBend, const int& deIndex);
  void makeCluster(PreClusters::PreClusterNBP& clNonBend, const int& deIndex);
  void makeCluster(PreClusters::PreClusterBP& clBend, const int& deIndex);
  void makeCluster(PreClusters::PreClusterBP& clBend, PreClusters::PreClusterBP& clBendNeigh, PreClusters::PreClusterNBP& clNonBend, const int& deIndex);

  std::vector<Cluster2D> mClusters; ///< list of clusters
  unsigned long int mNClusters = 0; ///< Number of clusters
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERIZER_H */
