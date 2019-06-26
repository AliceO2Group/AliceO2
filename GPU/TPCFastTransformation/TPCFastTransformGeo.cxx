// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file  TPCFastTransformGeo.cxx
/// \brief Implementation of TPCFastTransformGeo class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastTransformGeo.h"
#include "FlatObject.h"
#include "GPUCommonMath.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace GPUCA_NAMESPACE::gpu;

TPCFastTransformGeo::TPCFastTransformGeo()
{
  // Default Constructor: creates an empty uninitialized object
  double dAlpha = 2. * M_PI / (NumberOfSlicesA);
  for (int i = 0; i < NumberOfSlices; i++) {
    SliceInfo& s = mSliceInfos[i];
    double alpha = dAlpha * (i + 0.5);
    s.sinAlpha = sin(alpha);
    s.cosAlpha = cos(alpha);
  }
  mSliceInfos[NumberOfSlices] = SliceInfo{ 0.f, 0.f };

  for (int i = 0; i < MaxNumberOfRows + 1; i++) {
    mRowInfos[i] = RowInfo{ 0.f, -1, 0.f, 0.f, 0.f, 0.f };
  }
}

void TPCFastTransformGeo::startConstruction(int numberOfRows)
{
  /// Starts the construction procedure

  assert(numberOfRows >= 0 && numberOfRows < MaxNumberOfRows);

  mConstructionMask = ConstructionState::InProgress;
  mNumberOfRows = numberOfRows;

  mTPCzLengthA = 0.f;
  mTPCzLengthC = 0.f;
  mTPCalignmentZ = 0.f;
  mScaleVtoSVsideA = 0.f;
  mScaleVtoSVsideC = 0.f;
  mScaleSVtoVsideA = 0.f;
  mScaleSVtoVsideC = 0.f;

  for (int i = 0; i < MaxNumberOfRows; i++) {
    mRowInfos[i] = RowInfo{ 0.f, -1, 0.f, 0.f, 0.f, 0.f };
  }
}

void TPCFastTransformGeo::setTPCzLength(float tpcZlengthSideA, float tpcZlengthSideC)
{
  /// Sets TPC z length for both sides

  assert(mConstructionMask & ConstructionState::InProgress);
  assert((tpcZlengthSideA > 0.f) && (tpcZlengthSideC > 0.f));

  mTPCzLengthA = tpcZlengthSideA;
  mTPCzLengthC = tpcZlengthSideC;
  mScaleVtoSVsideA = 1. / tpcZlengthSideA;
  mScaleVtoSVsideC = 1. / tpcZlengthSideC;
  mScaleSVtoVsideA = tpcZlengthSideA;
  mScaleSVtoVsideC = tpcZlengthSideC;

  mConstructionMask |= ConstructionState::GeometryIsSet;
}

void TPCFastTransformGeo::setTPCalignmentZ(float tpcAlignmentZ)
{
  /// Sets the TPC alignment
  assert(mConstructionMask & ConstructionState::InProgress);

  mTPCalignmentZ = tpcAlignmentZ;
  mConstructionMask |= ConstructionState::AlignmentIsSet;
}

void TPCFastTransformGeo::setTPCrow(int iRow, float x, int nPads, float padWidth)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mNumberOfRows);
  assert(nPads > 1);
  assert(padWidth > 0.);

  // Make scaled U = area between centers of the first and the last pad

  // double uWidth = (nPads - 1) * padWidth;

  // Make scaled U = area between the geometrical sector borders

  const double sectorAngle = 2. * M_PI / NumberOfSlicesA;
  const double scaleXtoRowWidth = 2. * tan(0.5 * sectorAngle);
  double uWidth = x * scaleXtoRowWidth; // distance to the sector border

  RowInfo& row = mRowInfos[iRow];
  row.x = x;
  row.maxPad = nPads - 1;
  row.padWidth = padWidth;
  row.u0 = -uWidth / 2.;
  row.scaleUtoSU = 1. / uWidth;
  row.scaleSUtoU = uWidth;
}

void TPCFastTransformGeo::finishConstruction()
{
  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);     // construction in process
  assert(mConstructionMask & ConstructionState::GeometryIsSet);  // geometry is  set
  assert(mConstructionMask & ConstructionState::AlignmentIsSet); // alignment is  set

  for (int i = 0; i < mNumberOfRows; i++) { // all TPC rows are initialized
    assert(getRowInfo(i).maxPad > 0);
  }

  mConstructionMask = (unsigned int)ConstructionState::Constructed; // clear all other construction flags
}

void TPCFastTransformGeo::print() const
{
  /// Prints the geometry
#if !defined(GPUCA_GPUCODE)
  std::cout << "TPC Fast Transformation Geometry: " << std::endl;
  std::cout << "mNumberOfRows = " << mNumberOfRows << std::endl;
  std::cout << "mTPCzLengthA = " << mTPCzLengthA << std::endl;
  std::cout << "mTPCzLengthC = " << mTPCzLengthC << std::endl;
  std::cout << "mTPCalignmentZ = " << mTPCalignmentZ << std::endl;
  std::cout << "TPC Rows : " << std::endl;
  for (int i = 0; i < mNumberOfRows; i++) {
    std::cout << " tpc row " << i << ": x = " << mRowInfos[i].x << " maxPad = " << mRowInfos[i].maxPad << " padWidth = " << mRowInfos[i].padWidth << std::endl;
  }
#endif
}

int TPCFastTransformGeo::test(int slice, int row, float ly, float lz) const
{
  /// Check consistency of the class

  int error = 0;

  if (!isConstructed())
    error = -1;

  if (mNumberOfRows <= 0 || mNumberOfRows >= MaxNumberOfRows)
    error = -2;

  float lx = getRowInfo(row).x;
  float lx1 = 0.f, ly1 = 0.f, lz1 = 0.f;
  float gx = 0.f, gy = 0.f, gz = 0.f;

  convLocalToGlobal(slice, lx, ly, lz, gx, gy, gz);
  convGlobalToLocal(slice, gx, gy, gz, lx1, ly1, lz1);

  if (fabs(lx1 - lx) > 1.e-4 || fabs(ly1 - ly) > 1.e-4 || fabs(lz1 - lz) > 1.e-7) {
    std::cout << "Error local <-> global: x " << lx << " dx " << lx1 - lx << " y " << ly << " dy " << ly1 - ly << " z " << lz << " dz " << lz1 - lz << std::endl;
    error = -3;
  }
  float u = 0.f, v = 0.f;
  convLocalToUV(slice, ly, lz, u, v);
  convUVtoLocal(slice, u, v, ly1, lz1);

  if (fabs(ly1 - ly) + fabs(lz1 - lz) > 1.e-6) {
    std::cout << "Error local <-> UV: y " << ly << " dy " << ly1 - ly << " z " << lz << " dz " << lz1 - lz << std::endl;
    error = -4;
  }

  float su = 0.f, sv = 0.f;

  convUVtoScaledUV(slice, row, u, v, su, sv);

  if (su < 0.f || su > 1.f) {
    std::cout << "Error scaled U range: u " << u << " su " << su << std::endl;
    error = -5;
  }

  float u1 = 0.f, v1 = 0.f;
  convScaledUVtoUV(slice, row, su, sv, u1, v1);

  if (fabs(u1 - u) > 1.e-4 || fabs(v1 - v) > 1.e-4) {
    std::cout << "Error UV<->scaled UV: u " << u << " du " << u1 - u << " v " << v << " dv " << v1 - v << std::endl;
    error = -6;
  }

  float pad = 0.f;
  convUtoPad(row, u, pad);
  convPadToU(row, pad, u1);

  if (fabs(u1 - u) > 1.e-5) {
    std::cout << "Error U<->Pad: u " << u << " pad " << pad << " du " << u1 - u << std::endl;
    error = -7;
  }

#if !defined(GPUCA_GPUCODE)
  if (error != 0) {
    std::cout << "TPC Fast Transformation Geometry: Internal ERROR " << error << std::endl;
  }
#endif
  return error;
}

int TPCFastTransformGeo::test() const
{
  /// Check consistency of the class

  return test(2, 5, 10., 10.); // test at an arbitrary position
}
