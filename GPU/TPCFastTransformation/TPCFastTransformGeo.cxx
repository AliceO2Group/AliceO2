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

/// \file  TPCFastTransformGeo.cxx
/// \brief Implementation of TPCFastTransformGeo class
///
/// \author  Sergey Gorbunov <sergey.gorbunov@cern.ch>

#include "TPCFastTransformGeo.h"
#include "FlatObject.h"
#include "GPUCommonMath.h"
#include "GPUCommonLogger.h"

#if !defined(GPUCA_GPUCODE)
#include <iostream>
#endif

using namespace o2::gpu;

TPCFastTransformGeo::TPCFastTransformGeo()
{
  // Default Constructor: creates an empty uninitialized object
  double dAlpha = 2. * M_PI / (NumberOfSectorsA);
  for (int32_t i = 0; i < NumberOfSectors; i++) {
    SectorInfo& s = mSectorInfos[i];
    double alpha = dAlpha * (i + 0.5);
    s.sinAlpha = sin(alpha);
    s.cosAlpha = cos(alpha);
  }
  mSectorInfos[NumberOfSectors] = SectorInfo{};

  for (int32_t i = 0; i < MaxNumberOfRows + 1; i++) {
    mRowInfos[i] = RowInfo{};
  }
}

void TPCFastTransformGeo::startConstruction(int32_t numberOfRows)
{
  /// Starts the construction procedure

  assert(numberOfRows >= 0 && numberOfRows < MaxNumberOfRows);

  mConstructionMask = ConstructionState::InProgress;
  mNumberOfRows = numberOfRows;

  mTPCzLength = 0.f;

  for (int32_t i = 0; i < MaxNumberOfRows; i++) {
    mRowInfos[i] = RowInfo{};
  }
}

void TPCFastTransformGeo::setTPCzLength(float tpcZlength)
{
  /// Sets TPC z length for both sides

  assert(mConstructionMask & ConstructionState::InProgress);
  assert(tpcZlength > 0.f);

  mTPCzLength = tpcZlength;

  mConstructionMask |= ConstructionState::GeometryIsSet;
}

void TPCFastTransformGeo::setTPCrow(int32_t iRow, float x, int32_t nPads, float padWidth)
{
  /// Initializes a TPC row
  assert(mConstructionMask & ConstructionState::InProgress);
  assert(iRow >= 0 && iRow < mNumberOfRows);
  assert(nPads > 1);
  assert(padWidth > 0.);

  // Make scaled U = area between centers of the first and the last pad

  // double uWidth = (nPads - 1) * padWidth;

  // Make scaled U = area between the geometrical sector borders

  const double sectorAngle = 2. * M_PI / NumberOfSectorsA;
  const double scaleXtoRowWidth = 2. * tan(0.5 * sectorAngle);
  double uWidth = x * scaleXtoRowWidth; // distance to the sector border

  RowInfo& row = mRowInfos[iRow];
  row.x = x;
  row.maxPad = nPads - 1;
  row.padWidth = padWidth;
  row.yMin = -uWidth / 2.;
}

void TPCFastTransformGeo::finishConstruction()
{
  /// Finishes initialization: puts everything to the flat buffer, releases temporary memory

  assert(mConstructionMask & ConstructionState::InProgress);    // construction in process
  assert(mConstructionMask & ConstructionState::GeometryIsSet); // geometry is  set

  for (int32_t i = 0; i < mNumberOfRows; i++) { // all TPC rows are initialized
    assert(getRowInfo(i).maxPad > 0);
  }

  mConstructionMask = (uint32_t)ConstructionState::Constructed; // clear all other construction flags
}

void TPCFastTransformGeo::print() const
{
/// Prints the geometry
#if !defined(GPUCA_GPUCODE)
  LOG(info) << "TPC Fast Transformation Geometry: ";
  LOG(info) << "mNumberOfRows = " << mNumberOfRows;
  LOG(info) << "mTPCzLength = " << mTPCzLength;
  LOG(info) << "TPC Rows : ";
  for (int32_t i = 0; i < mNumberOfRows; i++) {
    LOG(info) << " tpc row " << i << ": x = " << mRowInfos[i].x << " maxPad = " << mRowInfos[i].maxPad << " padWidth = " << mRowInfos[i].padWidth;
  }
#endif
}

int32_t TPCFastTransformGeo::test(int32_t sector, int32_t row, float ly, float lz) const
{
  /// Check consistency of the class

  int32_t error = 0;

  if (!isConstructed()) {
    error = -1;
  }
  if (mNumberOfRows <= 0 || mNumberOfRows >= MaxNumberOfRows) {
    error = -2;
  }
  float lx = getRowInfo(row).x;
  float lx1 = 0.f, ly1 = 0.f, lz1 = 0.f;
  float gx = 0.f, gy = 0.f, gz = 0.f;

  convLocalToGlobal(sector, lx, ly, lz, gx, gy, gz);
  convGlobalToLocal(sector, gx, gy, gz, lx1, ly1, lz1);

  if (fabs(lx1 - lx) > 1.e-4 || fabs(ly1 - ly) > 1.e-4 || fabs(lz1 - lz) > 1.e-7) {
    LOG(info) << "Error local <-> global: x " << lx << " dx " << lx1 - lx << " y " << ly << " dy " << ly1 - ly << " z " << lz << " dz " << lz1 - lz;
    error = -3;
  }

  auto [pad, length] = convLocalToPadDriftLength(sector, 10, ly, lz);
  auto [ly2, lz2] = convPadDriftLengthToLocal(sector, 10, pad, length);

  if (fabs(ly2 - ly) + fabs(lz2 - lz) > 1.e-6) {
    LOG(info) << "Error local <-> UV: y " << ly << " dy " << ly2 - ly << " z " << lz << " dz " << lz2 - lz;
    error = -4;
  }

#if !defined(GPUCA_GPUCODE)
  if (error != 0) {
    LOG(info) << "TPC Fast Transformation Geometry: Internal ERROR " << error;
  }
#endif
  return error;
}

int32_t TPCFastTransformGeo::test() const
{
  /// Check consistency of the class

  return test(2, 5, 10., 10.); // test at an arbitrary position
}
