// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDTestingSimTools/HitFinder.h
/// \brief  Hit finder for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 March 2018

#ifndef O2_MID_HITFINDER_H
#define O2_MID_HITFINDER_H

#include "MathUtils/Cartesian3D.h"
#include "DataFormatsMID/Cluster2D.h"
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
  HitFinder(const GeometryTransformer& geoTrans);

  std::vector<int> getFiredDE(const Track& track, int chamber) const;
  std::vector<Cluster2D> getLocalPositions(const Track& track, int chamber, bool withUncertainty = false) const;

 private:
  Point3D<double> getIntersectInDefaultPlane(const Track& track, int chamber) const;
  Cluster2D getIntersect(const Track& track, int deId) const;
  int guessRPC(double yPos, int chamber) const;
  void addUncertainty(Cluster2D& cl, Track track) const;

  GeometryTransformer mGeometryTransformer; ///< Geometry transformer
  const double mTanTheta;                   ///< Tangent of the angle between y and z
  const double mCosTheta;                   ///< Cosinus of the beam angle
};
} // namespace mid
} // namespace o2

#endif /* O2_MID_HITFINDER_H */
