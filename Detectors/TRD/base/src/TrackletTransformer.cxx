// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
  // 2.221
  // mXAnode = mGeo->anodePos();
  // 3.35
  mXAnode = mGeo->cdrHght() + mGeo->camHght() / 2;
  // 2.5
  mXDrift = mGeo->cdrHght() - 0.5;
  mXtb0 = -100;

  // dummy values for testing. This will change in the future when values are pulled from CCDB
  mt0Correction = -0.1;
  mOldLorentzAngle = 0.16;
  mLorentzAngle = -0.14;
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

  // slightly modified TDP eq 16.1 (appended -1 to the end to account for MCM shared pads)
  double pad = float(position - (1 << (NBITSTRKLPOS - 1))) * GRANULARITYTRKLPOS + NCOLMCM * (4 * side + column) + 10.5 - 1;
  float y = padWidth * (pad - 72);

  return y;
}

float TrackletTransformer::calculateZ(int padrow)
{
  double rowPos = mPadPlane->getRowPos(padrow);
  double rowWidth = mPadPlane->getRowSize(padrow);
  double middleRowPos = mPadPlane->getRowPos(mPadPlane->getNrows() / 2);

  float z = rowPos - rowWidth / 2. - middleRowPos;

  return z;
}

float TrackletTransformer::calculateDy(int slope, double oldLorentzAngle, double lorentzAngle, double driftVRatio)
{
  double padWidth = mPadPlane->getWidthIPad();

  // temporary dummy value in cm/microsecond
  float vDrift = 1.56;
  float driftHeight = mGeo->cdrHght();

  // dy = slope * nTimeBins * padWidth * GRANULARITYTRKLSLOPE;
  // nTimeBins should be number of timebins in drift region. 1 timebin is 100 nanosecond
  double rawDy = slope * ((driftHeight / vDrift) / 0.1) * padWidth * GRANULARITYTRKLSLOPE;

  // driftDistance = 3.35
  float driftDistance = mGeo->cdrHght() + mGeo->camHght();

  float cmSlope = rawDy / driftDistance;

  double calibratedDy = rawDy - (TMath::Tan(lorentzAngle) * driftDistance);
  calibratedDy += (TMath::Tan(oldLorentzAngle) * driftDistance * driftVRatio) + cmSlope * (driftDistance * (1 - driftVRatio));

  // ALTERNATIVE METHOD

  // double x_anode_hit = driftDistance*driftVRatio/cmSlope;
  // double y_anode_hit = driftDistance*driftVRatio;

  // double x_Lorentz_drift_hit = TMath::Tan(oldLorentzAngle)*driftDistance*driftVRatio - TMath::Tan(lorentzAngle)*driftDistance;
  // double y_Lorentz_drift_hit = driftDistance*driftVRatio - driftDistance;

  // double Delta_x_Lorentz_drift_hit = x_anode_hit - x_Lorentz_drift_hit;
  // double Delta_y_Lorentz_drift_hit = y_anode_hit - y_Lorentz_drift_hit;
  // double impact_angle_rec = TMath::ATan2(Delta_y_Lorentz_drift_hit,Delta_x_Lorentz_drift_hit);

  // float calibrationShift = TMath::Tan(impact_angle_rec) * driftDistance;

  // LOG(info) << "ORIGINAL: " << calibratedDy;
  // LOG(info) << "ALTERNATIVE: " << rawDy + calibrationShift;

  return calibratedDy;
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
  uint64_t trackletWord = tracklet.getTrackletWord();
  uint64_t hcid = tracklet.getHCID();
  uint64_t padrow = tracklet.getPadRow();
  uint64_t column = tracklet.getColumn();
  // 0-2048 | units:pad-widths | granularity=1/75 (measured from center pad 10) 1024 is 0/center of pad 10
  uint64_t position = tracklet.getPosition();
  // 0-127 | units:pads/timebin | granularity=1/1000
  uint64_t slope = tracklet.getSlope();

  // calculate raw local chamber space point
  loadPadPlane(hcid);
  float x = getXDrift();
  float y = calculateY(hcid, column, position);
  float z = calculateZ(padrow);

  float dy = calculateDy(slope, mOldLorentzAngle, mLorentzAngle, mDriftVRatio);
  float calibratedX = calibrateX(x, mt0Correction);

  std::array<float, 3> sectorSpacePoint = transformL2T(hcid, std::array<double, 3>{x, y, z});

  LOG(debug) << "x: " << sectorSpacePoint[0] << " | "
             << "y: " << sectorSpacePoint[1] << " | "
             << "z: " << sectorSpacePoint[2];

  CalibratedTracklet calibratedTracklet = CalibratedTracklet(trackletWord, sectorSpacePoint[0], sectorSpacePoint[1], sectorSpacePoint[2], dy);

  return calibratedTracklet;
}
