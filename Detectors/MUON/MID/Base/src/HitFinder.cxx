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

/// \file   MID/Base/src/HitFinder.cxx
/// \brief  Implementation of the hit finder for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   14 March 2018

#include "MIDBase/HitFinder.h"

#include <cmath>
#include "MIDBase/DetectorParameters.h"
#include "MIDBase/GeometryParameters.h"

namespace o2
{
namespace mid
{
//______________________________________________________________________________
HitFinder::HitFinder(const GeometryTransformer& geoTrans)
  : mGeometryTransformer(geoTrans),
    mTanTheta(std::tan((90. - geoparams::BeamAngle) * std::atan(1) / 45.)),
    mCosTheta(std::cos(geoparams::BeamAngle * std::atan(1) / 45.))
{
  /// default constructor
}

//______________________________________________________________________________
math_utils::Point3D<double> HitFinder::getIntersectInDefaultPlane(const Track& track, int chamber) const
{
  double defaultZ = geoparams::DefaultChamberZ[chamber];
  double linePar = ((track.getPositionZ() - defaultZ) * mTanTheta - track.getPositionY()) /
                   (track.getDirectionY() - track.getDirectionZ() * mTanTheta);
  math_utils::Point3D<double> point;
  point.SetX(track.getPositionX() + linePar * track.getDirectionX());
  point.SetY(track.getPositionY() + linePar * track.getDirectionY());
  point.SetZ(track.getPositionZ() + linePar * track.getDirectionZ());
  return point;
}

//______________________________________________________________________________
Cluster HitFinder::getIntersect(const Track& track, int deId) const
{
  math_utils::Point3D<float> localPoint = mGeometryTransformer.globalToLocal(deId, track.getPositionX(), track.getPositionY(), track.getPositionZ());

  math_utils::Vector3D<float> localDirection = mGeometryTransformer.globalToLocal(deId, math_utils::Vector3D<float>(track.getDirectionX(), track.getDirectionY(), track.getDirectionZ()));
  Track localTrack;
  localTrack.setPosition(localPoint.x(), localPoint.y(), localPoint.z());
  localTrack.setDirection(localDirection.x() / localDirection.z(), localDirection.y() / localDirection.z(), 1.);

  localTrack.propagateToZ(0);
  Cluster cluster;
  cluster.deId = deId;
  cluster.xCoor = localTrack.getPositionX();
  cluster.yCoor = localTrack.getPositionY();
  return cluster;
}

//______________________________________________________________________________
int HitFinder::guessRPC(double yPos, int chamber) const
{
  return (int)(yPos / (2. * geoparams::getRPCHalfHeight(chamber)) + 4.5);
}

//______________________________________________________________________________
std::vector<int> HitFinder::getFiredDE(const Track& track, int chamber) const
{
  std::vector<int> deIdList;
  math_utils::Point3D<double> defPos = getIntersectInDefaultPlane(track, chamber);
  double xPos = defPos.x();
  double xErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarX));
  double yPos = defPos.y() / mCosTheta;
  double yErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarY));
  double positions[3] = {yPos, yPos - yErr, yPos + yErr};
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
      deIdList.push_back(detparams::getDEId(true, chamber, rpc));
      if (xPos - xErr < 0) {
        deIdList.push_back(detparams::getDEId(false, chamber, rpc));
      }
    } else {
      deIdList.push_back(detparams::getDEId(false, chamber, rpc));
      if (xPos + xErr > 0) {
        deIdList.push_back(detparams::getDEId(true, chamber, rpc));
      }
    }
  }

  return deIdList;
}

//______________________________________________________________________________
std::vector<Cluster> HitFinder::getLocalPositions(const Track& track, int chamber, bool withUncertainties) const
{
  std::vector<int> deIdList = getFiredDE(track, chamber);

  std::vector<Cluster> points;
  for (auto& deId : deIdList) {
    Cluster cl = getIntersect(track, deId);
    int rpc = detparams::getRPCLine(deId);
    double hl = geoparams::getRPCHalfLength(chamber, rpc);
    double hh = geoparams::getRPCHalfHeight(chamber);
    if (cl.xCoor < -hl || cl.xCoor > hl) {
      continue;
    }
    if (cl.yCoor < -hh || cl.yCoor > hh) {
      continue;
    }
    if (withUncertainties) {
      addUncertainty(cl, track);
    }
    points.emplace_back(cl);
  }

  return points;
}

//______________________________________________________________________________
void HitFinder::addUncertainty(Cluster& cl, Track track) const
{
  auto globalPos = mGeometryTransformer.localToGlobal(cl.deId, cl.xCoor, cl.yCoor);
  track.propagateToZ(globalPos.z());
  cl.xErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarX));
  cl.yErr = std::sqrt(track.getCovarianceParameter(Track::CovarianceParamIndex::VarY));
}

} // namespace mid
} // namespace o2
