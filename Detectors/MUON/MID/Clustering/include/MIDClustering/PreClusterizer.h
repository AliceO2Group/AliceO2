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

/// \file   MIDClustering/PreClusterizer.h
/// \brief  Pre-cluster reconstruction algorithm for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   05 July 2018

#ifndef O2_MID_PRECLUSTERIZER_H
#define O2_MID_PRECLUSTERIZER_H

#include <unordered_map>
#include <unordered_set>
#include <vector>
#include <gsl/gsl>
#include "MIDBase/Mapping.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDClustering/PreCluster.h"

namespace o2
{
namespace mid
{
/// Pre-clustering algorithm for MID
class PreClusterizer
{
 public:
  /// Builds the pre-clusters from the strip patterns in the event
  /// \param stripPatterns Vector of strip patterns per column
  /// \param accumulate Flag to decide if one needs to reset the output preclusters at each event
  void process(gsl::span<const ColumnData> stripPatterns, bool accumulate = false);

  /// Builds the pre-clusters from the strip patterns in the timeframe
  /// \param stripPatterns Vector of strip patterns per column
  /// \param rofRecords RO frame records
  void process(gsl::span<const ColumnData> stripPatterns, gsl::span<const ROFRecord> rofRecords);

  /// Gets the vector of reconstructed pre-clusters
  const std::vector<PreCluster>& getPreClusters() { return mPreClusters; }

  /// Gets the vector of pre-clusters RO frame records
  const std::vector<ROFRecord>& getROFRecords() { return mROFRecords; }

 private:
  struct PatternStruct {
    uint8_t deId = 0;                  ///< Detection element ID
    int firedColumns = 0;              ///< Fired columns
    std::array<ColumnData, 7> columns; ///< Array of strip patterns
  };

  /// Fills the mpDE structure with fired pads
  /// \param stripPatterns Vector of strip patterns per column
  /// \return true if stripPatterns is not empty
  bool loadPatterns(gsl::span<const ColumnData>& stripPatterns);

  /// Builds the pre-clusters for the Bending Plane
  /// \param de structure containing the ordered strip patterns for one Detection Element
  void preClusterizeBP(PatternStruct& de);

  /// Builds the pre-clusters for the Non-Bending Plane
  /// \param de structure containing the ordered strip patterns for one Detection Element
  void preClusterizeNBP(PatternStruct& de);

  Mapping mMapping;                                  //!< Mapping
  std::unordered_map<uint8_t, PatternStruct> mMpDEs; //!< Internal mapping
  std::unordered_set<uint8_t> mActiveDEs;            //!< List of active detection elements for event
  std::vector<PreCluster> mPreClusters;              ///< List of pre-clusters
  std::vector<ROFRecord> mROFRecords;                ///< List of pre-clusters RO frame records
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_PRECLUSTERIZER_H */
