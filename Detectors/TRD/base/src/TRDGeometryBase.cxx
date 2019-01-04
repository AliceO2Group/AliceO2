// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "TRDBase/TRDGeometryBase.h"

using namespace o2::trd;

//_____________________________________________________________________________
GPUd() int TRDGeometryBase::getStack(float z, int layer) const
{
  //
  // Reconstruct the chamber number from the z position and layer number
  //
  // The return function has to be protected for positiveness !!
  //

  if ((layer < 0) || (layer >= kNlayer))
    return -1;

  int istck = kNstack;
  float zmin = 0.0;
  float zmax = 0.0;

  do {
    istck--;
    if (istck < 0)
      break;
    const TRDPadPlane* pp = getPadPlane(layer, istck);
    zmax = pp->getRow0();
    int nrows = pp->getNrows();
    zmin = zmax - 2 * pp->getLengthOPad() - (nrows - 2) * pp->getLengthIPad() - (nrows - 1) * pp->getRowSpacing();
  } while ((z < zmin) || (z > zmax));

  return istck;
}

//_____________________________________________________________________________
GPUd() bool TRDGeometryBase::isOnBoundary(int det, float y, float z, float eps) const
{
  //
  // Checks whether position is at the boundary of the sensitive volume
  //

  int ly = getLayer(det);
  if ((ly < 0) || (ly >= kNlayer))
    return true;

  int stk = getStack(det);
  if ((stk < 0) || (stk >= kNstack))
    return true;

  const TRDPadPlane* pp = &mPadPlanes[getDetectorSec(ly, stk)];
  if (!pp)
    return true;

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
