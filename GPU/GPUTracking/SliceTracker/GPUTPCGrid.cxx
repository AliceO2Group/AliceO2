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

#ifndef assert
#include <assert.h>
#endif

void GPUTPCGrid::CreateEmpty()
{
	//Create an empty grid
	fYMin = 0.f;
	fYMax = 1.f;
	fZMin = 0.f;
	fZMax = 1.f;

	fNy = 0;
	fNz = 0;
	fN = 0;

	fStepYInv = 1.f;
	fStepZInv = 1.f;
}

GPUd() void GPUTPCGrid::Create(float yMin, float yMax, float zMin, float zMax, float sy, float sz)
{
	//* Create the grid
	fYMin = yMin;
	fZMin = zMin;

	fStepYInv = 1.f / sy;
	fStepZInv = 1.f / sz;

	fNy = static_cast<unsigned int>((yMax - fYMin) * fStepYInv + 1.f);
	fNz = static_cast<unsigned int>((zMax - fZMin) * fStepZInv + 1.f);

	fN = fNy * fNz;

	fYMax = fYMin + fNy * sy;
	fZMax = fZMin + fNz * sz;
}

GPUd() int GPUTPCGrid::GetBin(float Y, float Z) const
{
	//* get the bin pointer
	const int yBin = static_cast<int>((Y - fYMin) * fStepYInv);
	const int zBin = static_cast<int>((Z - fZMin) * fStepZInv);
	const int bin = zBin * fNy + yBin;
#ifndef GPUCA_GPUCODE
	assert(bin >= 0);
	assert(bin < static_cast<int>(fN));
#endif
	return bin;
}

GPUd() int GPUTPCGrid::GetBinBounded(float Y, float Z) const
{
	//* get the bin pointer
	const int yBin = static_cast<int>((Y - fYMin) * fStepYInv);
	const int zBin = static_cast<int>((Z - fZMin) * fStepZInv);
	const int bin = zBin * fNy + yBin;
	if (bin < 0) return 0;
	if (bin >= static_cast<int>(fN)) return fN - 1;
	return bin;
}

GPUd() void GPUTPCGrid::GetBin(float Y, float Z, int *const bY, int *const bZ) const
{
	//* get the bin pointer

	int bbY = (int) ((Y - fYMin) * fStepYInv);
	int bbZ = (int) ((Z - fZMin) * fStepZInv);

	if (bbY < 0)
		bbY = 0;
	else if (bbY >= (int) fNy)
		bbY = fNy - 1;
	if (bbZ < 0)
		bbZ = 0;
	else if (bbZ >= (int) fNz)
		bbZ = fNz - 1;
	*bY = (unsigned int) bbY;
	*bZ = (unsigned int) bbZ;
}

GPUd() void GPUTPCGrid::GetBinArea(float Y, float Z, float dy, float dz, int &bin, int &ny, int &nz) const
{
	Y -= fYMin;
	int by = (int) ((Y - dy) * fStepYInv);
	ny = (int) ((Y + dy) * fStepYInv) - by;
	Z -= fZMin;
	int bz = (int) ((Z - dz) * fStepZInv);
	nz = (int) ((Z + dz) * fStepZInv) - bz;
	if (by < 0)
		by = 0;
	else if (by >= (int) fNy)
		by = fNy - 1;
	if (bz < 0)
		bz = 0;
	else if (bz >= (int) fNz)
		bz = fNz - 1;
	if (by + ny >= (int) fNy) ny = fNy - 1 - by;
	if (bz + nz >= (int) fNz) nz = fNz - 1 - bz;
	bin = bz * fNy + by;
}
