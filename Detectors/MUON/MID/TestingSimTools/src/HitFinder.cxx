// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/TestingSimTools/src/HitFinder.cxx
/// \brief  Implementation of the hit finder for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 March 2018

#include "MIDTestingSimTools/HitFinder.h"

#include <cmath>
#include "MIDBase/Constants.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
HitFinder::HitFinder(const GeometryTransformer& geoTrans)
  : mGeometryTransformer(geoTrans),
    mTanTheta(std::tan((90. - Constants::sBeamAngle) * std::atan(1) / 45.)),
    mCosTheta(std::cos(Constants::sBeamAngle * std::atan(1) / 45.))
{
  /// default constructor
}

//______________________________________________________________________________
Point3D<double> HitFinder::getIntersectInDefaultPlane(const Track& track, int chamber) const
{
  /// Get the intersection point in the default chamber plane
  double defaultZ = Constants::sDefaultChamberZ[chamber];
  double linePar = ((track.getPositionZ() - defaultZ) * mTanTheta - track.getPositionY()) /
                   (track.getDirectionY() - track.getDirectionZ() * mTanTheta);
  Point3D<double> point;
  point.SetX(track.getPositionX() + linePar * track.getDirectionX());
  point.SetY(track.getPositionY() + linePar * track.getDirectionY());
  point.SetZ(track.getPositionZ() + linePar * track.getDirectionZ());
  return point;
}

//______________________________________________________________________________
Point3D<float> HitFinder::getIntersect(const Track& track, int deId) const
{
  /// Get the intersection point in the specified detection elements
  /// The point is expressed in local coordinates
  Point3D<float> localPoint = mGeometryTransformer.globalToLocal(deId, track.getPositionX(), track.getPositionY(), track.getPositionZ());

  // Track localTrack(track);
  // localTrack.propagateToZ(localTrack.getPosition().z() + 20.);
  // Point3D<float> localPoint2 = mGeometryTransformer.globalToLocal(deId, localTrack.getPosition());
  // localTrack.setPosition(localPoint.x(), localPoint.y(), localPoint.z());
  // float dZ = localPoint2.z() - localPoint.z();
  // localTrack.setDirection((localPoint2.x() - localPoint.x()) / dZ, (localPoint2.y() - localPoint.y()) / dZ, 1.);

  Vector3D<float> localDirection = mGeometryTransformer.globalToLocal(deId, Vector3D<float>(track.getDirectionX(), track.getDirectionY(), track.getDirectionZ()));
  Track localTrack;
  localTrack.setPosition(localPoint.x(), localPoint.y(), localPoint.z());
  localTrack.setDirection(localDirection.x() / localDirection.z(), localDirection.y() / localDirection.z(), 1.);

  localTrack.propagateToZ(0);
  Point3D<float> point(localTrack.getPositionX(), localTrack.getPositionY(), localTrack.getPositionZ());
  return std::move(point);
}

//______________________________________________________________________________
int HitFinder::guessRPC(double yPos, int chamber) const
{
  /// Guesses the RPC form the y position
  return (int)(yPos / (2. * Constants::getRPCHalfHeight(chamber)) + 4.5);
}

//______________________________________________________________________________
std::vector<int> HitFinder::getFiredDE(const Track& track, int chamber) const
{
  /// Gets the potentially fired detection elements
  /// @param track MID track
  /// @param chamber Chamber ID (0-3)
  /// @return Vector with the list of the detection element IDs potentially fired
  std::vector<int> deIdList;
  Point3D<double> defPos = getIntersectInDefaultPlane(track, chamber);
  double xPos = defPos.x();
  double xErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarX));
  double yPos = defPos.y() / mCosTheta;
  double yErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarY));
  double positions[3] = { yPos, yPos - yErr, yPos + yErr };
  int centerRpc = -1;
  for (int ipos = 0; ipos < 3; ++ipos) {
    int rpc = guessRPC(positions[ipos], chamber);
    if (rpc < 0 || rpc > 8) {
      continue;
    }
    if (ipos == 0) {
      centerRpc = rpc;
    } else if (rpc == centerRpc) {
      continue;
    }
    if (xPos > 0) {
      deIdList.push_back(Constants::getDEId(true, chamber, rpc));
      if (xPos - xErr < 0) {
        deIdList.push_back(Constants::getDEId(false, chamber, rpc));
      }
    } else {
      deIdList.push_back(Constants::getDEId(false, chamber, rpc));
      if (xPos + xErr > 0) {
        deIdList.push_back(Constants::getDEId(true, chamber, rpc));
      }
    }
  }

  return deIdList;
}

//______________________________________________________________________________
std::vector<std::pair<int, Point3D<float>>> HitFinder::getLocalPositions(const Track& track, int chamber) const
{
  /// Gets the list of fired Points in local coordinates
  /// @param track MID track
  /// @param chamber Chamber ID (0-3)
  /// @return Vector with the pairs of detection element Ids and intersection point
  std::vector<int> deIdList = getFiredDE(track, chamber);

  std::vector<std::pair<int, Point3D<float>>> points;
  for (auto& deId : deIdList) {
    Point3D<float> point = getIntersect(track, deId);
    int rpc = deId % 9;
    double hl = Constants::getRPCHalfLength(chamber, rpc);
    double hh = Constants::getRPCHalfHeight(chamber);
    if (point.x() < -hl || point.x() > hl) {
      continue;
    }
    if (point.y() < -hh || point.y() > hh) {
      continue;
    }
    points.emplace_back(std::pair<int, Point3D<float>>(deId, point));
  }

  return points;
}

} // namespace mid
} // namespace o2
