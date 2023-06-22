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


#include "TRDQC/CoordinateTransformer.h"
#include "TRDBase/Geometry.h"


using namespace o2::trd;

CoordinateTransformer::CoordinateTransformer() 
: mGeo(o2::trd::Geometry::instance()) 
{
  mGeo->createPadPlaneArray();
}

std::array<float, 3> CoordinateTransformer::Local2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det)%6, (det/6)%5);

  // array<double,3> rct;

  double iPadLen = padPlane->getLengthIPad();
  double oPadLen = padPlane->getLengthOPad();
  int nRows = padPlane->getNrows();

  double lengthCorr = padPlane->getLengthOPad()/padPlane->getLengthIPad();

  // calculate position based on inner pad length
  rct[0] = - z / padPlane->getLengthIPad() + padPlane->getNrows()/2;

  // correct row for outer pad rows
  if (rct[0] <= 1.0) {
    rct[0] = 1.0 - (1.0-rct[0])*lengthCorr;
  }

  if (rct[0] >= double(nRows-1)) {
    rct[0] = double(nRows-1) + (rct[0] - double(nRows-1))*lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  if ( rct[0] < 0.0 || rct[0] > double(nRows) ) {
    std::cout << "ERROR: hit with z=" << z << ", padrow " << rct[0]
          << " outside of chamber" << std::endl;
  }

  // simple conversion of pad / local y coordinate
  // ignore different width of outer pad
  rct[1] = y / padPlane->getWidthIPad() + 144./2.;

  // time coordinate
  if (x<-0.35) {
    // drift region
    rct[2] = mT0 - (x+0.35) / (mVdrift/10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  rct[1] += (x + 0.35) * mExB;
  return rct;
}

std::array<float, 3> CoordinateTransformer::OrigLocal2RCT(int det, float x, float y, float z)
{
  std::array<float, 3> rct;

  auto padPlane = mGeo->getPadPlane((det)%6, (det/6)%5);

  // array<double,3> rct;

  double iPadLen = padPlane->getLengthIPad();
  double oPadLen = padPlane->getLengthOPad();
  int nRows = padPlane->getNrows();

  double lengthCorr = padPlane->getLengthOPad()/padPlane->getLengthIPad();

  // calculate position based on inner pad length
  rct[0] = - z / padPlane->getLengthIPad() + padPlane->getNrows()/2;

  // correct row for outer pad rows
  if (rct[0] <= 1.0) {
    rct[0] = 1.0 - (1.0-rct[0])*lengthCorr;
  }

  if (rct[0] >= double(nRows-1)) {
    rct[0] = double(nRows-1) + (rct[0] - double(nRows-1))*lengthCorr;
  }

  // sanity check: is the padrow coordinate reasonable?
  if ( rct[0] < 0.0 || rct[0] > double(nRows) ) {
    std::cout << "ERROR: hit with z=" << z << ", padrow " << rct[0]
          << " outside of chamber" << std::endl;
  }

  // simple conversion of pad / local y coordinate
  // ignore different width of outer pad
  rct[1] = y / padPlane->getWidthIPad() + 144./2.;

  // time coordinate
  if (x<-0.35) {
    // drift region
    rct[2] = mT0 - (x+0.35) / (mVdrift/10.0);
  } else {
    // anode region: very rough guess
    rct[2] = mT0 - 1.0 + fabs(x);
  }

  return rct;
}



// std::ostream& operator<<(std::ostream& os, const ChamberSpacePoint& p)
// {
//   os << "( " << std::setprecision(5) << p.getX() 
//      << " / " << std::setprecision(5) << p.getY() 
//      << " / " << std::setprecision(6) << p.getZ() << ") <-> ["
//      << p.getDetector() << "." << p.getPadRow() 
//      << " pad " << std::setprecision(5) << p.getPadCol() << "]";
//   return os;
// }
