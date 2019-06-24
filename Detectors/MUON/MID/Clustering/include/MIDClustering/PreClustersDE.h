// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDClustering/PreClustersDE.h
/// \brief  Structure with pre-clusters in the MID detection element
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   04 September 2018

#ifndef O2_MID_PRECLUSTERSDE_H
#define O2_MID_PRECLUSTERSDE_H

#include <array>
#include <vector>
#include "MIDBase/MpArea.h"

namespace o2
{
namespace mid
{

// Internal pre-cluster structure for MID clustering
class PreClustersDE
{
 public:
  struct NBP {
    size_t index;               ///< Index of input pre-cluster
    int paired;                 /// Is paired flag
    std::array<MpArea, 7> area; ///< 2D area containing the PreCluster per column
  };

  struct BP {
    size_t index; ///< Index of input pre-cluster
    int paired;   /// Is paired flag
    MpArea area;  ///< 2D area containing the PreCluster per column
  };

  /// Gets the pre-cluster in the NBP
  NBP& getPreClusterNBP(int idx) { return mPreClustersNBP[idx]; }

  /// Gets the pre-cluster in the NBP (const version)
  const NBP& getPreClusterNBP(int idx) const { return mPreClustersNBP[idx]; }

  /// Gets pre-cluster in the BP
  BP& getPreClusterBP(int icolumn, int idx) { return mPreClustersBP[icolumn][idx]; }

  /// Gets pre-cluster in the BP (const version)
  const BP&
    getPreClusterBP(int icolumn, int idx) const { return mPreClustersBP[icolumn][idx]; }

  /// Gets the number of pre-clusters in the NBP
  size_t getNPreClustersNBP() const { return mPreClustersNBP.size(); }

  /// Gets the number of pre-clusters in the BP
  size_t getNPreClustersBP(int icolumn) const { return mPreClustersBP[icolumn].size(); }

  /// Sets the detection element ID
  void setDEId(int deIndex) { mDEId = deIndex; }

  /// Gets the detection element ID
  int getDEId() const { return mDEId; }

  std::vector<int> getNeighbours(int icolumn, int idx) const;

  /// Gets the vector of pre-clusters in the NBP
  std::vector<NBP>& getPreClustersNBP() { return mPreClustersNBP; }

  /// Gets the vector of pre-clusters in the BP in column icolumn
  std::vector<BP>& getPreClustersBP(int icolumn) { return mPreClustersBP[icolumn]; }

  bool init();
  void reset();

 private:
  int mDEId = 99;                                ///< Detection element ID
  std::vector<NBP> mPreClustersNBP;              ///< list of PreClusters in the NBP in each DE
  std::array<std::vector<BP>, 7> mPreClustersBP; ///< list of PreClusters in the BP in each DE per column
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERSDE_H */
