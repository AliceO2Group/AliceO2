// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDClustering/PreClusters.h
/// \brief  Pre-clusters structure for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 September 2018

#ifndef O2_MID_PRECLUSTERS_H
#define O2_MID_PRECLUSTERS_H

#include <vector>
#include "MIDBase/Mapping.h"

namespace o2
{
namespace mid
{

/// Pre-cluster structure for MID
class PreClusters
{
 public:
  PreClusters();
  virtual ~PreClusters() = default;

  struct PreClusterNBP {
    int firstColumn;            ///< First fired column
    int lastColumn;             ///< Last fired column
    int paired;                 ///< Flag to check if PreCliuster was paired
    std::array<MpArea, 7> area; ///< 2D area containing the PreCluster per column
  };

  struct PreClusterBP {
    int column;  ///< Fired column
    int paired;  ///< Flag to check if PreCluster was paired
    MpArea area; ///< 2D area containing the PreCluster per column
  };

  /// Gets the pre-cluster in the NBP
  PreClusterNBP& getPreClusterNBP(int idx) { return mPreClustersNBP[idx]; }

  /// Gets the pre-cluster in the NBP (const version)
  const PreClusterNBP& getPreClusterNBP(int idx) const { return mPreClustersNBP[idx]; }

  /// Gets pre-cluster in the BP
  PreClusterBP&
    getPreClusterBP(int icolumn, int idx) { return mPreClustersBP[icolumn][idx]; }

  /// Gets pre-cluster in the BP (const version)
  const PreClusterBP&
    getPreClusterBP(int icolumn, int idx) const { return mPreClustersBP[icolumn][idx]; }

  /// Gets the number of pre-clusters in the NBP
  int getNPreClustersNBP() const { return mNPreClustersNBP; }

  /// Gets the number of pre-clusters in the BP
  int getNPreClustersBP(int icolumn) const { return mNPreClustersBP[icolumn]; }

  /// Sets the detection element ID
  void setDEId(int deIndex) { mDEId = deIndex; }

  /// Gets the detection element ID
  int getDEId() const { return mDEId; }

  std::vector<int> getNeighbours(int icolumn, int idx);

  PreClusterNBP* nextPreClusterNBP();
  PreClusterBP* nextPreClusterBP(int icolumn);

  bool init();
  void reset();

 private:
  int mDEId = 99;                     ///< Detection element ID
  int mNPreClustersNBP = 0;           ///< number of PreClusters in the NBP in each DE
  std::array<int, 7> mNPreClustersBP; ///< number of PreClusters in the BP in each DE per column

  std::vector<PreClusterNBP> mPreClustersNBP; ///< list of PreClusters in the NBP in each DEı
  std::array<std::vector<PreClusterBP>, 7>
    mPreClustersBP; ///< list of PreClusters in the BP in each DE per column
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERS_H */
