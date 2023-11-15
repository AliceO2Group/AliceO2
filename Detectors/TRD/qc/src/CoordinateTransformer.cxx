// Copyright 2019-2023 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDQC/CoordinateTransformer.h"
#include <TMath.h>
#include "DataFormatsTRD/Constants.h"
#include "TRDBase/Geometry.h"

using namespace o2::trd;

float ChamberSpacePoint::getMCMChannel(int mcmcol) const
{
  float c = float(mcmcol * o2::trd::constants::NCOLMCM + o2::trd::constants::NADCMCM) - getPadCol();
  if (c < 0.0 || c > o2::trd::constants::NADCMCM) {
    return -1;
  } else {
    return c;
  }
}

bool ChamberSpacePoint::isInMCM(int detector, int padrow, int mcmcol) const
{
  if (detector != mDetector || padrow != mPadrow) {
    return false;
  } else {
    // calculate the channel number in the MCM for the
    float c = getMCMChannel(mcmcol);
    if (c < 0.0 || c > o2::trd::constants::NADCMCM) {
      return false;
    } else {
      return true;
    }
  }
}

CoordinateTransformer::CoordinateTransformer()
  : mGeo(o2::trd::Geometry::instance())
{
  mGeo->createPadPlaneArray();
}

std::array<float, 3> CoordinateTransformer::Local2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det) % 6, (det / 6) % 5);

  // for the z-coordinate, we combine the row number and offset from the padPlane into a single float
  int row = padPlane->getPadRow(z);
  if (row == 0 || row == padPlane->getNrows() - 1) {
    rct[0] = float(row) + padPlane->getPadRowOffsetROC(row, z) / padPlane->getLengthOPad();
  } else {
    rct[0] = float(row) + padPlane->getPadRowOffsetROC(row, z) / padPlane->getLengthIPad();
  }

  // the y-coordinate is calculated directly by the padPlane object
  rct[1] = padPlane->getPad(y, z);

  // we calculate the time coordinate by hand
  if (x < -0.35) {
    // in drift region:
    //   account for offset between anode and cathode wires: add 0.35
    //   convert drift velocity to from cm/us to cm/timebin
    rct[2] = mT0 - (x + 0.35) / (mVdrift / 10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  // Correct for Lorentz angle, but only in the drift region. ExB in the anode region causes a small offset (ca. 0.1
  // pads) that is constant for all clusters in the drift region.
  rct[1] += (x + 0.35) * mExB;
  return rct;
}

std::array<float, 3> CoordinateTransformer::OrigLocal2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det) % 6, (det / 6) % 5);

  // array<double,3> rct;

  double iPadLen = padPlane->getLengthIPad();
  double oPadLen = padPlane->getLengthOPad();
  int nRows = padPlane->getNrows();

  double lengthCorr = padPlane->getLengthOPad() / padPlane->getLengthIPad();

  // calculate position based on inner pad length
  rct[0] = -z / padPlane->getLengthIPad() + padPlane->getNrows() / 2;

  // correct row for outer pad rows
  if (rct[0] <= 1.0) {
    rct[0] = 1.0 - (1.0 - rct[0]) * lengthCorr;
  }

  if (rct[0] >= double(nRows - 1)) {
    rct[0] = double(nRows - 1) + (rct[0] - double(nRows - 1)) * lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  if (rct[0] < 0.0 || rct[0] > double(nRows)) {
    std::cout << "ERROR: hit with z=" << z << ", padrow " << rct[0]
              << " outside of chamber" << std::endl;
  }

  // simple conversion of pad / local y coordinate
  // ignore different width of outer pad
  rct[1] = y / padPlane->getWidthIPad() + 144. / 2.;

  // time coordinate
  if (x < -0.35) {
    // drift region
    rct[2] = mT0 - (x + 0.35) / (mVdrift / 10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  return rct;
}

o2::trd::ChamberSpacePoint CoordinateTransformer::MakeSpacePoint(o2::trd::Hit& hit)
{
  float x = hit.getLocalT();
  float y = hit.getLocalC();
  float z = hit.getLocalR();
  auto rct = Local2RCT(hit.GetDetectorID(), x, y, z);
  return o2::trd::ChamberSpacePoint(hit.GetTrackID(), hit.GetDetectorID(), x, y, y, rct, hit.isFromDriftRegion());
}

namespace o2::trd
{
std::ostream& operator<<(std::ostream& os, const ChamberSpacePoint& p)
{
  int sector = p.getDetector() / 30;
  int stack = (p.getDetector() % 30) / 6;
  int layer = p.getDetector() % 6;

  os << "( " << std::setprecision(5) << p.getX()
     << " / " << std::setprecision(5) << p.getY()
     << " / " << std::setprecision(6) << p.getZ() << ") <-> ["
     << sector << "_" << stack << "_" << layer << " (" << p.getDetector() << ")"
     << " row " << p.getPadRow()
     << " pad " << std::setprecision(5) << p.getPadCol() << "]";
  return os;
}
}; // namespace o2::trd