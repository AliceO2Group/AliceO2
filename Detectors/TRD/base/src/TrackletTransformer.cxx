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

#include "DetectorsBase/GeometryManager.h"
#include "TRDBase/TrackletTransformer.h"
#include "TMath.h"

using namespace o2::trd;
using namespace o2::trd::constants;

TrackletTransformer::TrackletTransformer()
{
  o2::base::GeometryManager::loadGeometry();
  mGeo = Geometry::instance();
  mGeo->createPadPlaneArray();
  mGeo->createClusterMatrixArray();

  // 3 cm
  mXCathode = mGeo->cdrHght();
  // 3.35
  mXAnode = mGeo->cdrHght() + mGeo->camHght() / 2;
  // 5mm below cathode plane to reduce error propogation from tracklet fit and driftV
  mXDrift = mGeo->cdrHght() - 0.5;
  mXtb0 = -100;

  // dummy values for testing. This will change in the future when values are pulled from CCDB
  mt0Correction = -0.279;
  mLorentzAngle = 0.14;
  mDriftVRatio = 1.1;
}

void TrackletTransformer::loadPadPlane(int hcid)
{
  int detector = hcid / 2;
  int stack = mGeo->getStack(detector);
  int layer = mGeo->getLayer(detector);
  mPadPlane = mGeo->getPadPlane(layer, stack);
}

float TrackletTransformer::calculateY(int hcid, int column, int position)
{
  double padWidth = mPadPlane->getWidthIPad();
  int side = hcid % 2;

  // the position calculated in TRAPsim is a signed integer
  int positionUnsigned = 0;
  if (position & (1 << (NBITSTRKLPOS - 1))) {
    positionUnsigned = -((~(position - 1)) & ((1 << NBITSTRKLPOS) - 1));
  } else {
    positionUnsigned = position & ((1 << NBITSTRKLPOS) - 1);
  }
  positionUnsigned += 1 << (NBITSTRKLPOS - 1); // shift such that positionUnsigned = 1 << (NBITSTRKLPOS - 1) corresponds to the MCM center

  // slightly modified TDP eq 16.1 (appended -1 to the end to account for MCM shared pads)
  double pad = float(positionUnsigned - (1 << (NBITSTRKLPOS - 1))) * GRANULARITYTRKLPOS + NCOLMCM * (4 * side + column) + 10. - 1.;
  float y = padWidth * (pad - 72);

  return y;
}

float TrackletTransformer::calculateZ(int padrow)
{
  double rowPos = mPadPlane->getRowPos(padrow);
  double rowSize = mPadPlane->getRowSize(padrow);
  double middleRowPos = mPadPlane->getRowPos(mPadPlane->getNrows() / 2);

  return rowPos - rowSize / 2. - middleRowPos;
}

float TrackletTransformer::calculateDy(int slope, double lorentzAngle, double driftVRatio)
{
  double padWidth = mPadPlane->getWidthIPad();

  // temporary dummy value in cm/microsecond
  float vDrift = 1.5464f;

  int slopeSigned = 0;
  if (slope & (1 << (NBITSTRKLSLOPE - 1))) {
    slopeSigned = -((~(slope - 1)) & ((1 << NBITSTRKLSLOPE) - 1));
  } else {
    slopeSigned = slope & ((1 << NBITSTRKLSLOPE) - 1);
  }

  // dy = slope * nTimeBins * padWidth * GRANULARITYTRKLSLOPE;
  // nTimeBins should be number of timebins in drift region. 1 timebin is 100 nanosecond
  double rawDy = slopeSigned * ((mXCathode / vDrift) * 10.) * padWidth * GRANULARITYTRKLSLOPE;

  // NOTE: check what drift height is used in calibration code to ensure consistency
  // NOTE: check sign convention of Lorentz angle
  // NOTE: confirm the direction in which vDrift is measured/determined. Is it in x or in direction of drift?
  double lorentzCorrection = TMath::Tan(lorentzAngle) * mXAnode;

  // assuming angle in Bailhache, fig. 4.17 would be positive in our calibration code
  double calibratedDy = rawDy - lorentzCorrection;

  return rawDy;
}

float TrackletTransformer::calibrateX(double x, double t0Correction)
{
  return x += t0Correction;
}

std::array<float, 3> TrackletTransformer::transformL2T(int hcid, std::array<double, 3> point)
{
  int detector = hcid / 2;
  auto transformationMatrix = mGeo->getMatrixT2L(detector);

  ROOT::Math::Impl::Transform3D<double>::Point localPoint(point[0], point[1], point[2]);
  auto gobalPoint = transformationMatrix ^ localPoint;

  return {(float)gobalPoint.x(), (float)gobalPoint.y(), (float)gobalPoint.z()};
}

CalibratedTracklet TrackletTransformer::transformTracklet(Tracklet64 tracklet)
{
  uint64_t hcid = tracklet.getHCID();
  uint64_t padrow = tracklet.getPadRow();
  uint64_t column = tracklet.getColumn();
  uint64_t position = tracklet.getPosition();
  uint64_t slope = tracklet.getSlope();

  // calculate raw local chamber space point
  loadPadPlane(hcid);
  float x = getXDrift();
  float y = calculateY(hcid, column, position);
  float z = calculateZ(padrow);

  float dy = calculateDy(slope, mLorentzAngle, mDriftVRatio);
  float calibratedX = calibrateX(x, mt0Correction);
  // NOTE: Correction to y position based on x calibration NOT YET implemented. Need t0.

  std::array<float, 3> sectorSpacePoint = transformL2T(hcid, std::array<double, 3>{calibratedX, y, z});

  LOG(debug) << "x: " << sectorSpacePoint[0] << " | "
             << "y: " << sectorSpacePoint[1] << " | "
             << "z: " << sectorSpacePoint[2];

  return CalibratedTracklet(sectorSpacePoint[0], sectorSpacePoint[1], sectorSpacePoint[2], dy);
}

double TrackletTransformer::getTimebin(double x)
{
  // calculate timebin from x position within chamber
  // calibration parameters need to be extracted from CCDB in the future
  double vDrift = 1.5625; // in cm/us
  double t0 = 4.0;        // time (in timebins) of start of drift region

  double timebin;
  // x = 0 at anode plane and points toward pad plane.
  if (x < -mGeo->camHght() / 2) {
    // drift region
    timebin = t0 - (x + mGeo->camHght() / 2) / (vDrift * 0.1);
  } else {
    // anode region: very rough guess
    timebin = t0 - 1.0 + fabs(x);
  }

  return timebin;
}
