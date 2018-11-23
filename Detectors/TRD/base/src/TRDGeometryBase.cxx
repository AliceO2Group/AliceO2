// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <FairLogger.h>

#include "TRDBase/TRDGeometryBase.h"
#include "TRDBase/TRDPadPlane.h"

using namespace o2::trd;

//_____________________________________________________________________________
int TRDGeometryBase::getDetectorSec(int layer, int stack) const
{
  //
  // Convert plane / stack into detector number for one single sector
  //

  return (layer + stack * kNlayer);
}

//_____________________________________________________________________________
int TRDGeometryBase::getDetector(int layer, int stack, int sector) const
{
  //
  // Convert layer / stack / sector into detector number
  //

  return (layer + stack * kNlayer + sector * kNlayer * kNstack);
}

//_____________________________________________________________________________
int TRDGeometryBase::getLayer(int det) const
{
  //
  // Reconstruct the layer number from the detector number
  //

  return ((int)(det % kNlayer));
}

//_____________________________________________________________________________
int TRDGeometryBase::getStack(int det) const
{
  //
  // Reconstruct the stack number from the detector number
  //

  return ((int)(det % (kNlayer * kNstack)) / kNlayer);
}

//_____________________________________________________________________________
int TRDGeometryBase::getStack(float z, int layer) const
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
    TRDPadPlane* pp = getPadPlane(layer, istck);
    zmax = pp->getRow0();
    int nrows = pp->getNrows();
    zmin = zmax - 2 * pp->getLengthOPad() - (nrows - 2) * pp->getLengthIPad() - (nrows - 1) * pp->getRowSpacing();
  } while ((z < zmin) || (z > zmax));

  return istck;
}

//_____________________________________________________________________________
TRDPadPlane* TRDGeometryBase::getPadPlane(int layer, int stack) const
{
  //
  // Returns the pad plane for a given plane <pl> and stack <st> number
  //

  int ipp = getDetectorSec(layer, stack);
  return &mPadPlaneArray[ipp];
}

//_____________________________________________________________________________
int TRDGeometryBase::getRowMax(int layer, int stack, int /*sector*/) const
{
  //
  // Returns the number of rows on the pad plane
  //

  return getPadPlane(layer, stack)->getNrows();
}

//_____________________________________________________________________________
int TRDGeometryBase::getColMax(int layer) const
{
  //
  // Returns the number of rows on the pad plane
  //

  return getPadPlane(layer, 0)->getNcols();
}

//_____________________________________________________________________________
float TRDGeometryBase::getRow0(int layer, int stack, int /*sector*/) const
{
  //
  // Returns the position of the border of the first pad in a row
  //

  return getPadPlane(layer, stack)->getRow0();
}

//_____________________________________________________________________________
float TRDGeometryBase::getCol0(int layer) const
{
  //
  // Returns the position of the border of the first pad in a column
  //

  return getPadPlane(layer, 0)->getCol0();
}

//_____________________________________________________________________________
bool TRDGeometryBase::isHole(int /*la*/, int st, int se) const
{
  //
  // Checks for holes in front of PHOS
  //

  if (((se == 13) || (se == 14) || (se == 15)) && (st == 2)) {
    return true;
  }

  return false;
}

//_____________________________________________________________________________
bool TRDGeometryBase::isOnBoundary(int det, float y, float z, float eps) const
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

  TRDPadPlane* pp = &mPadPlaneArray[getDetectorSec(ly, stk)];
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
