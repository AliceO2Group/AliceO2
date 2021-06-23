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

#include "TRDBase/GeometryBase.h"

using namespace o2::trd;
using namespace o2::trd::constants;

//_____________________________________________________________________________
GPUd() int GeometryBase::getStack(float z, int layer) const
{
  //
  // Reconstruct the chamber number from the z position and layer number
  //
  // The return function has to be protected for positiveness !!
  //

  if ((layer < 0) || (layer >= NLAYER)) {
    return -1;
  }

  int istck = NSTACK;
  float zmin = 0.0;
  float zmax = 0.0;

  do {
    istck--;
    if (istck < 0) {
      break;
    }
    const PadPlane* pp = getPadPlane(layer, istck);
    zmax = pp->getRow0();
    int nrows = pp->getNrows();
    zmin = zmax - 2 * pp->getLengthOPad() - (nrows - 2) * pp->getLengthIPad() - (nrows - 1) * pp->getRowSpacing();
  } while ((z < zmin) || (z > zmax));

  return istck;
}

//_____________________________________________________________________________
GPUd() bool GeometryBase::isOnBoundary(int det, float y, float z, float eps) const
{
  //
  // Checks whether position is at the boundary of the sensitive volume
  //

  int ly = getLayer(det);
  if ((ly < 0) || (ly >= NLAYER)) {
    return true;
  }

  int stk = getStack(det);
  if ((stk < 0) || (stk >= NSTACK)) {
    return true;
  }

  const PadPlane* pp = &mPadPlanes[getDetectorSec(ly, stk)];
  if (!pp) {
    return true;
  }

  float max = pp->getRow0();
  int n = pp->getNrows();
  float min = max - 2 * pp->getLengthOPad() - (n - 2) * pp->getLengthIPad() - (n - 1) * pp->getRowSpacing();
  if (z < min + eps || z > max - eps) {
    // printf("z : min[%7.2f (%7.2f)] %7.2f max[(%7.2f) %7.2f]\n", min, min+eps, z, max-eps, max);
    return true;
  }
  min = pp->getCol0();
  n = pp->getNcols();
  max = min + 2 * pp->getWidthOPad() + (n - 2) * pp->getWidthIPad() + (n - 1) * pp->getColSpacing();
  if (y < min + eps || y > max - eps) {
    // printf("y : min[%7.2f (%7.2f)] %7.2f max[(%7.2f) %7.2f]\n", min, min+eps, y, max-eps, max);
    return true;
  }

  return false;
}
