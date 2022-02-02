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

/// \file   MIDBase/HitFinder.h
/// \brief  Hit finder for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 March 2018

#ifndef O2_MID_HITFINDER_H
#define O2_MID_HITFINDER_H

#include "MathUtils/Cartesian.h"
#include "DataFormatsMID/Cluster.h"
#include "DataFormatsMID/Track.h"
#include "MIDBase/GeometryTransformer.h"

namespace o2
{
namespace mid
{
/// Class to find the impact point of a track on the chamber
class HitFinder
{
 public:
  /// Constructor
  /// \param geoTrans Geometry transformer
  HitFinder(const GeometryTransformer& geoTrans);

  /// Default destructor
  ~HitFinder() = default;

  /// Returns the geometry transformer
  const GeometryTransformer& getGeometryTransformer() const { return mGeometryTransformer; }

  /// Gets the potentially fired detection elements
  /// \param track MID track
  /// \param chamber Chamber ID (0-3)
  /// \return Vector with the list of the detection element IDs potentially fired
  std::vector<int> getFiredDE(const Track& track, int chamber) const;

  /// Gets the list of track-deId intersection in local coordinates, expressed as clusters
  /// \param track MID track
  /// \param chamber Chamber ID (0-3)
  /// \param withUncertainties Add also the extrapolated uncertainties to the local point
  /// \return Vector of intersection cluster(s) with the detection element(s)
  std::vector<Cluster> getLocalPositions(const Track& track, int chamber, bool withUncertainties = false) const;

 private:
  /// Get the intersection point in the default chamber plane
  /// \param track MID track
  /// \param chamber chamber plane (from 0 to 3)
  /// \return Intersection of the track and the deID
  math_utils::Point3D<double> getIntersectInDefaultPlane(const Track& track, int chamber) const;

  /// Get the intersection point in the specified detection elements
  /// The point is expressed in local coordinates
  /// \param track MID track
  /// \param deId Detection element ID
  /// \return Intersection of the track and the deID, expressed as a cluster with no uncertainties
  Cluster getIntersect(const Track& track, int deId) const;

  /// Guesses the RPC form the y position
  /// \param yPos y position
  /// \param chamber chamber plane (from 0 to 3)
  /// \return the possible detection element ID
  int guessRPC(double yPos, int chamber) const;

  /// Adds uncertainties to cluster
  /// \param cl Cluster to be modified
  /// \param track MID track in local coordinates
  void addUncertainty(Cluster& cl, Track track) const;

  GeometryTransformer mGeometryTransformer; ///< Geometry transformer
  const double mTanTheta;                   ///< Tangent of the angle between y and z
  const double mCosTheta;                   ///< Cosine of the beam angle
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_HITFINDER_H */
