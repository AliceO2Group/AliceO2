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

/// \file   MIDTracking/HitMapBuilder.h
/// \brief  Utility to build the MID track hit maps
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   10 December 2021

#ifndef O2_MID_HITMAPBUILDER_H
#define O2_MID_HITMAPBUILDER_H

#include <vector>
#include <unordered_set>
#include <gsl/gsl>
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/ROFRecord.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/GeometryTransformer.h"
#include "MIDBase/HitFinder.h"
#include "MIDBase/Mapping.h"

namespace o2
{
namespace mid
{
/// Hit map builder for MID
class HitMapBuilder
{
 public:
  /// Constructor
  /// \param geoTrans Geometry transformer
  HitMapBuilder(const GeometryTransformer& geoTrans);

  /// Default destructor
  ~HitMapBuilder() = default;

  /// Builds the track infos
  /// \param track Reconstructed track: it will be modified
  /// \param clusters gsl::span of associated clusters (in local coordinates)
  /// \return TrackInfo object with the track hit map and other information
  void buildTrackInfo(Track& track, gsl::span<const Cluster> clusters) const;

  /// Builds the track infos for the tracks in the vector
  /// \param tracks Vector of reconstructed tracks: it will be modified
  /// \param clusters gsl::span of associated clusters (in local coordinates)
  /// \return TrackInfo object with the track hit map and other information
  void process(std::vector<Track>& tracks, gsl::span<const Cluster> clusters) const;

 private:
  /// Checks if the track crossed the same element
  /// \param fired Vector of fired elements
  /// \param nonFired Vector of non-fired elements
  /// \return true if the track crossed the same element
  bool crossCommonElement(const std::vector<int>& fired, const std::vector<int>& nonFired) const;

  /// Returns the efficiency flag
  /// \param firedRPCLines Vector of fired RPC lines
  /// \param nonFiredRPCLines Vector of non-fired RPC lines
  /// \param firedLocIds Vector of fired Local board Ids
  /// \param nonFiredLocIds Vector of non-fired Local board Ids
  /// \return the efficiency flag
  int getEffFlag(const std::vector<int>& firedRPCLines, const std::vector<int>& nonFiredRPCLines, const std::vector<int>& firedLocIds, const std::vector<int>& nonFiredLocIds) const;

  /// Function to extract the local board ID of the cluster
  /// \param xp cluster x coordinate
  /// \param yp cluster y coordinate
  /// \param deId cluster Detection Element ID
  /// \return the Local board ID
  int getLocId(double xp, double yp, uint8_t deId) const;

  Mapping mMapping;     ///< Mapping
  HitFinder mHitFinder; ///< Hit finder
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_HITMAPBUILDER_H */
