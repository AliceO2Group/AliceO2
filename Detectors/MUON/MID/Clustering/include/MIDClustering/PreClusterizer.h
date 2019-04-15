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
/// \brief  Pre-cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018

#ifndef O2_MID_PRECLUSTERIZER_H
#define O2_MID_PRECLUSTERIZER_H

#include <unordered_map>
#include <vector>
#include <gsl/gsl>
#include "MIDBase/Mapping.h"
#include "DataFormatsMID/ColumnData.h"
#include "MIDClustering/PreCluster.h"

namespace o2
{
namespace mid
{
/// Pre-clustering algorithm for MID
class PreClusterizer
{
 public:
  bool init();
  bool process(gsl::span<const ColumnData> stripPatterns);

  /// Gets the vector of reconstructed pre-clusters
  const std::vector<PreCluster>& getPreClusters() { return mPreClusters; }

 private:
  struct PatternStruct {
    int deId;                          ///< Detection element ID
    int firedColumns;                  ///< Fired columns
    std::array<ColumnData, 7> columns; ///< Array of strip patterns
  };

  bool loadPatterns(gsl::span<const ColumnData>& stripPatterns);
  void reset();

  void preClusterizeBP(PatternStruct& de);
  void preClusterizeNBP(PatternStruct& de);

  Mapping mMapping;                              ///< Mapping
  std::unordered_map<int, PatternStruct> mMpDEs; ///< Internal mapping
  std::unordered_map<int, bool> mActiveDEs;      ///< List of active detection elements for event
  std::vector<PreCluster> mPreClusters;          ///< List of pre-clusters
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERIZER_H */
