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
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/TrackletTransformer.h"
#include "TMath.h"

using namespace o2::trd;
using namespace o2::trd::constants;

void TrackletTransformer::init()
{
  mGeo = Geometry::instance();
  mGeo->createPadPlaneArray();
  mGeo->createClusterMatrixArray();

  // 3.35 cm
  mXAnode = mGeo->cdrHght() + mGeo->camHght() / 2;
}

float TrackletTransformer::calculateY(int hcid, int column, int position, const PadPlane* padPlane) const
{
  double padWidth = padPlane->getWidthIPad();
  int side = hcid % 2;
  position += 1 << (NBITSTRKLPOS - 1); // shift such that position = 1 << (NBITSTRKLPOS - 1) corresponds to the MCM center
  // slightly modified TDP eq 16.1 (appended -1 to the end to account for MCM shared pads)
  double pad = float(position - (1 << (NBITSTRKLPOS - 1))) * GRANULARITYTRKLPOS + NCOLMCM * (4 * side + column) + 10. - 1.;
  float y = padWidth * (pad - 72);

  return y;
}

float TrackletTransformer::calculateZ(int padrow, const PadPlane* padPlane) const
{
  double rowPos = padPlane->getRowPos(padrow);
  double rowSize = padPlane->getRowSize(padrow);
  double middleRowPos = padPlane->getRowPos(padPlane->getNrows() / 2);

  return rowPos - rowSize / 2. - middleRowPos;
}

float TrackletTransformer::calculateDy(int detector, int slope, const PadPlane* padPlane) const
{
  double padWidth = padPlane->getWidthIPad();

  float vDrift = mCalVdriftExB->getVdrift(detector);
  float exb = mCalVdriftExB->getExB(detector);

  // dy = slope * nTimeBins * padWidth * GRANULARITYTRKLSLOPE;
  // nTimeBins should be number of timebins in drift region. 1 timebin is 100 nanosecond
  double rawDy = slope * ((mGeo->cdrHght() / vDrift) * 10.) * padWidth * GRANULARITYTRKLSLOPE / ADDBITSHIFTSLOPE;

  // NOTE: check what drift height is used in calibration code to ensure consistency
  // NOTE: check sign convention of Lorentz angle
  // NOTE: confirm the direction in which vDrift is measured/determined. Is it in x or in direction of drift?
  double lorentzCorrection = TMath::Tan(exb) * mXAnode;

  // assuming angle in Bailhache, fig. 4.17 would be positive in our calibration code
  double calibratedDy = rawDy - lorentzCorrection;

  return calibratedDy;
}

float TrackletTransformer::calibrateX(double x) const
{
  // Hard-coded value will be replaced once t0 calibration is available from CCDB
  float t0Correction = -0.279;
  return x += t0Correction;
}

std::array<float, 3> TrackletTransformer::transformL2T(int detector, std::array<double, 3> point) const
{
  auto transformationMatrix = mGeo->getMatrixT2L(detector);

  ROOT::Math::Impl::Transform3D<double>::Point localPoint(point[0], point[1], point[2]);
  auto gobalPoint = transformationMatrix ^ localPoint;

  return {(float)gobalPoint.x(), (float)gobalPoint.y(), (float)gobalPoint.z()};
}

CalibratedTracklet TrackletTransformer::transformTracklet(Tracklet64 tracklet, bool trackingFrame) const
{
  auto detector = tracklet.getDetector();
  auto hcid = tracklet.getHCID();
  auto padrow = tracklet.getPadRow();
  auto column = tracklet.getColumn();
  int position;
  int slope;
  if (mApplyXOR) {
    position = tracklet.getPosition() ^ 0x80;
    if (position & (1 << (constants::NBITSTRKLPOS - 1))) {
      position = -((~(position - 1)) & ((1 << constants::NBITSTRKLPOS) - 1));
    }
    slope = tracklet.getSlope() ^ 0x80;
    if (slope & (1 << (constants::NBITSTRKLSLOPE - 1))) {
      slope = -((~(slope - 1)) & ((1 << constants::NBITSTRKLSLOPE) - 1));
    }
  } else {
    position = tracklet.getPositionBinSigned();
    slope = tracklet.getSlopeBinSigned();
  }

  // calculate raw local chamber space point
  const auto padPlane = mGeo->getPadPlane(detector);

  // 5mm below cathode plane to reduce error propogation from tracklet fit and driftV
  float x = mGeo->cdrHght() - 0.5;
  float y = calculateY(hcid, column, position, padPlane);
  float z = calculateZ(padrow, padPlane);

  float dy = calculateDy(detector, slope, padPlane);

  float calibratedX = calibrateX(x);

  // NOTE: Correction to y position based on x calibration NOT YET implemented. Need t0.
  if (trackingFrame) {
    std::array<float, 3> sectorSpacePoint = transformL2T(detector, std::array<double, 3>{calibratedX, y, z});
    LOG(debug) << "x: " << sectorSpacePoint[0] << " | "
               << "y: " << sectorSpacePoint[1] << " | "
               << "z: " << sectorSpacePoint[2];
    return CalibratedTracklet(sectorSpacePoint[0], sectorSpacePoint[1], sectorSpacePoint[2], dy);
  } else {
    return CalibratedTracklet(calibratedX, y, z, dy); // local frame
  }
}

double TrackletTransformer::getTimebin(int detector, double x) const
{
  // calculate timebin from x position within chamber
  float vDrift = mCalVdriftExB->getVdrift(detector);
  double t0 = 4.0; // time (in timebins) of start of drift region

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
