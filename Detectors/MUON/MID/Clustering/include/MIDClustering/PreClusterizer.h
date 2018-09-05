// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDClustering/PreClusterizer.h
/// \brief  Pre-Cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018

#ifndef O2_MID_PRECLUSTERIZER_H
#define O2_MID_PRECLUSTERIZER_H

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
/// Pre-clustering algorithm for MID
class PreClusterizer
{
 public:
  PreClusterizer();
  virtual ~PreClusterizer() = default;

  PreClusterizer(const PreClusterizer&) = delete;
  PreClusterizer& operator=(const PreClusterizer&) = delete;
  PreClusterizer(PreClusterizer&&) = delete;
  PreClusterizer& operator=(PreClusterizer&&) = delete;

  bool init();
  bool process(const std::vector<ColumnData>& stripPatterns);

  /// Gets the array of reconstructes pre-clusters
  std::vector<PreClusters>& getPreClusters() { return mPreClusters; }

  /// Gets the number of reconstructed pre-clusters
  unsigned long int getNPreClusters() { return mNPreClusters; }

 private:
  struct PatternStruct {
    int deId;                          ///< Detection element ID
    int firedColumns;                  ///< Fired columns
    std::array<ColumnData, 7> columns; ///< Array of strip patterns
  };

  bool loadPatterns(const std::vector<ColumnData>& stripPatterns);
  void reset();

  void preClusterizeBP(PatternStruct& de, PreClusters& pcs);
  void preClusterizeNBP(PatternStruct& de, PreClusters& pcs);

  PreClusters& nextPreCluster();

  // bool buildListOfNeighbours(int icolumn, int lastColumn, std::vector<std::vector<std::pair<int, int>>>& neighbours, bool skipPaired = false, int currentList = 0);

  Mapping mMapping;                              ///< Mapping
  std::unordered_map<int, PatternStruct> mMpDEs; ///< Internal mapping
  std::vector<PreClusters> mPreClusters;         ///< List of pre-clusters
  int mNPreClusters;                             ///< Number of pre-clusters
  std::unordered_map<int, bool> mActiveDEs;      ///< List of active detection elements for event
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERIZER_H */
