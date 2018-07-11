// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Clustering/src/Clusterizer.h
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
  bool process(const std::vector<ColumnData>& stripPatterns);

  /// Gets the array of reconstructes clusters
  const std::vector<Cluster2D>& getClusters() { return mClusters; }

  /// Gets the number of reconstructed clusters
  unsigned long int getNClusters() { return mNClusters; }

 private:
  struct PatternStruct {
    int deId;                          ///< Detection element ID
    int firedColumns;                  ///< Fired columns
    std::array<ColumnData, 7> columns; ///< Array of strip patterns
  };

  struct PreCluster {
    int firstColumn;            ///< First fired column
    int lastColumn;             ///< Last fired column
    int paired;                 ///< Flag to check if PreCliuster was paired
    std::array<MpArea, 7> area; ///< 2D area containing the PreCluster per column
  };

  bool loadPatterns(const std::vector<ColumnData>& stripPatterns);
  void reset();

  void preClusterizeBP(PatternStruct& de);
  void preClusterizeNBP(PatternStruct& de);
  PreCluster* nextPreCluster(int icolumn);

  bool buildListOfNeighbours(int icolumn, int lastColumn, std::vector<std::vector<PreCluster*>>& neighbours,
                             bool skipPaired = false, int currentList = 0);

  Cluster2D& nextCluster();
  void makeClusters(const int& deIndex);
  void makeCluster(PreCluster& clBend, PreCluster& clNonBend, const int& deIndex);
  void makeCluster(std::vector<PreCluster*> pcBlist, const int& deIndex, PreCluster* clNonBend = nullptr);

  Mapping mMapping;                              ///< Mapping
  std::unordered_map<int, PatternStruct> mMpDEs; ///< internal mapping

  std::array<int, 8> mNPreClusters; ///< number of PreClusters in each DE per column (last column is the NBP)
  std::array<std::vector<PreCluster>, 8>
    mPreClusters; ///< list of PreClusters in each DE per column (last column is the NBP)

  std::unordered_map<int, bool> mActiveDEs; ///< List of active detection elements for event
  std::vector<Cluster2D> mClusters;         ///< list of clusters
  unsigned long int mNClusters;             ///< Number of clusters
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_CLUSTERIZER_H */
