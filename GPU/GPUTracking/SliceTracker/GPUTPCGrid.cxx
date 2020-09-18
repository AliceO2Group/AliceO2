// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGrid.cxx
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#include "GPUTPCGrid.h"
#include "GPUCommonMath.h"
using namespace GPUCA_NAMESPACE::gpu;

#if !defined(assert) && !defined(GPUCA_GPUCODE)
#include <cassert>
#endif

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCGrid)::CreateEmpty()
{
  // Create an empty grid
  mYMin = 0.f;
  mYMax = 1.f;
  mZMin = 0.f;
  mZMax = 1.f;

  mNy = 1;
  mNz = 1;
  mN = 1;

  mStepYInv = 1.f;
  mStepZInv = 1.f;
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCGrid)::Create(float yMin, float yMax, float zMin, float zMax, int ny, int nz)
{
  //* Create the grid
  mYMin = yMin;
  mZMin = zMin;

  float sy = CAMath::Max((yMax + 0.1f - yMin) / ny, GPUCA_MIN_BIN_SIZE);
  float sz = CAMath::Max((zMax + 0.1f - zMin) / nz, GPUCA_MIN_BIN_SIZE);

  mStepYInv = 1.f / sy;
  mStepZInv = 1.f / sz;

  mNy = ny;
  mNz = nz;

  mN = mNy * mNz;

  mYMax = mYMin + mNy * sy;
  mZMax = mZMin + mNz * sz;
}

MEM_CLASS_PRE()
GPUd() int MEM_LG(GPUTPCGrid)::GetBin(float Y, float Z) const
{
  //* get the bin pointer
  const int yBin = static_cast<int>((Y - mYMin) * mStepYInv);
  const int zBin = static_cast<int>((Z - mZMin) * mStepZInv);
  const int bin = zBin * mNy + yBin;
#ifndef GPUCA_GPUCODE
  assert(bin >= 0);
  assert(bin < static_cast<int>(mN));
#endif
  return bin;
}

MEM_CLASS_PRE()
GPUd() int MEM_LG(GPUTPCGrid)::GetBinBounded(float Y, float Z) const
{
  //* get the bin pointer
  const int yBin = static_cast<int>((Y - mYMin) * mStepYInv);
  const int zBin = static_cast<int>((Z - mZMin) * mStepZInv);
  int bin = zBin * mNy + yBin;
  if (bin >= static_cast<int>(mN)) {
    bin = mN - 1;
  }
  if (bin < 0) {
    bin = 0;
  }
  return bin;
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCGrid)::GetBin(float Y, float Z, int* const bY, int* const bZ) const
{
  //* get the bin pointer

  int bbY = (int)((Y - mYMin) * mStepYInv);
  int bbZ = (int)((Z - mZMin) * mStepZInv);

  if (bbY >= (int)mNy) {
    bbY = mNy - 1;
  }
  if (bbY < 0) {
    bbY = 0;
  }
  if (bbZ >= (int)mNz) {
    bbZ = mNz - 1;
  }
  if (bbZ < 0) {
    bbZ = 0;
  }

  *bY = (unsigned int)bbY;
  *bZ = (unsigned int)bbZ;
}

MEM_CLASS_PRE()
GPUd() void MEM_LG(GPUTPCGrid)::GetBinArea(float Y, float Z, float dy, float dz, int& bin, int& ny, int& nz) const
{
  Y -= mYMin;
  int by = (int)((Y - dy) * mStepYInv);
  ny = (int)((Y + dy) * mStepYInv) - by;
  Z -= mZMin;
  int bz = (int)((Z - dz) * mStepZInv);
  nz = (int)((Z + dz) * mStepZInv) - bz;
  if (by >= (int)mNy) {
    by = mNy - 1;
  }
  if (by < 0) {
    by = 0;
  }
  if (bz >= (int)mNz) {
    bz = mNz - 1;
  }
  if (bz < 0) {
    bz = 0;
  }
  if (by + ny >= (int)mNy) {
    ny = mNy - 1 - by;
  }
  if (bz + nz >= (int)mNz) {
    nz = mNz - 1 - bz;
  }
  bin = bz * mNy + by;
}
