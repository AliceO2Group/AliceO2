// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGrid.h
/// \author Sergey Gorbunov, Ivan Kisel, David Rohr

#ifndef GPUTPCGRID_H
#define GPUTPCGRID_H

#include "GPUTPCDef.h"

namespace GPUCA_NAMESPACE
{
namespace gpu
{
/**
 * @class GPUTPCGrid
 *
 * 2-dimensional grid of pointers.
 * pointers to (y,z)-like objects are assigned to the corresponding grid bin
 * used by GPUTPCTracker to speed-up the hit operations
 * grid axis are named Z,Y to be similar to TPC row coordinates.
 */
MEM_CLASS_PRE()
class GPUTPCGrid
{
 public:
  GPUd() void CreateEmpty();
  GPUd() void Create(float yMin, float yMax, float zMin, float zMax, int ny, int nz);

  GPUd() int GetBin(float Y, float Z) const;
  /**
 * returns -1 if the row is empty == no hits
 */
  GPUd() int GetBinBounded(float Y, float Z) const;
  GPUd() void GetBin(float Y, float Z, int* const bY, int* const bZ) const;
  GPUd() void GetBinArea(float Y, float Z, float dy, float dz, int& bin, int& ny, int& nz) const;

  GPUd() unsigned int N() const { return mN; }
  GPUd() unsigned int Ny() const { return mNy; }
  GPUd() unsigned int Nz() const { return mNz; }
  GPUd() float YMin() const { return mYMin; }
  GPUd() float YMax() const { return mYMax; }
  GPUd() float ZMin() const { return mZMin; }
  GPUd() float ZMax() const { return mZMax; }
  GPUd() float StepYInv() const { return mStepYInv; }
  GPUd() float StepZInv() const { return mStepZInv; }

 private:
  friend class GPUTPCNeighboursFinder;

  unsigned int mNy; //* N bins in Y
  unsigned int mNz; //* N bins in Z
  unsigned int mN;  //* total N bins
  float mYMin;      //* minimal Y value
  float mYMax;      //* maximal Y value
  float mZMin;      //* minimal Z value
  float mZMax;      //* maximal Z value
  float mStepYInv;  //* inverse bin size in Y
  float mStepZInv;  //* inverse bin size in Z
};
} // namespace gpu
} // namespace GPUCA_NAMESPACE

#endif // GPUTPCGRID_H
