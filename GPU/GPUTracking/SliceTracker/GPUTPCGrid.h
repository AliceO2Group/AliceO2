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

/**
 * @class GPUTPCGrid
 *
 * 2-dimensional grid of pointers.
 * pointers to (y,z)-like objects are assigned to the corresponding grid bin
 * used by GPUTPCTracker to speed-up the hit operations
 * grid axis are named Z,Y to be similar to TPC row coordinates.
 */
class GPUTPCGrid
{
  public:
	void CreateEmpty();
	GPUd() void Create(float yMin, float yMax, float zMin, float zMax, float sy, float sz);

	GPUd() int GetBin(float Y, float Z) const;
	/**
     * returns -1 if the row is empty == no hits
     */
	GPUd() int GetBinBounded(float Y, float Z) const;
	GPUd() void GetBin(float Y, float Z, int *const bY, int *const bZ) const;
	GPUd() void GetBinArea(float Y, float Z, float dy, float dz, int &bin, int &ny, int &nz) const;

	GPUd() unsigned int N() const { return fN; }
	GPUd() unsigned int Ny() const { return fNy; }
	GPUd() unsigned int Nz() const { return fNz; }
	GPUd() float YMin() const { return fYMin; }
	GPUd() float YMax() const { return fYMax; }
	GPUd() float ZMin() const { return fZMin; }
	GPUd() float ZMax() const { return fZMax; }
	GPUd() float StepYInv() const { return fStepYInv; }
	GPUd() float StepZInv() const { return fStepZInv; }

  private:
	unsigned int fNy; //* N bins in Y
	unsigned int fNz; //* N bins in Z
	unsigned int fN;  //* total N bins
	float fYMin;      //* minimal Y value
	float fYMax;      //* maximal Y value
	float fZMin;      //* minimal Z value
	float fZMax;      //* maximal Z value
	float fStepYInv;  //* inverse bin size in Y
	float fStepZInv;  //* inverse bin size in Z
};

#endif //GPUTPCGRID_H
