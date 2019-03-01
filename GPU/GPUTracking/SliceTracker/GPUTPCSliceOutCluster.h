// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutCluster.h
/// \author Sergey Gorbunov, David Rohr

#ifndef GPUTPCSLICEOUTCLUSTER_H
#define GPUTPCSLICEOUTCLUSTER_H

#include "GPUTPCDef.h"

/**
 * @class GPUTPCSliceOutCluster
 * GPUTPCSliceOutCluster class contains clusters which are assigned to slice tracks.
 * It is used to send the data from TPC slice trackers to the GlobalMerger
 */
class GPUTPCSliceOutCluster
{
  public:
	GPUhd() void Set(unsigned int id, unsigned char row, unsigned char flags, unsigned short amp, float x, float y, float z)
	{
		fRow = row;
		fFlags = flags;
		fAmp = amp;
		fId = id;
		fX = x;
		fY = y;
		fZ = z;
	}

	GPUhd() float GetX() const { return fX; }
	GPUhd() float GetY() const { return fY; }
	GPUhd() float GetZ() const { return fZ; }
	GPUhd() unsigned int GetId() const { return fId; }
	GPUhd() unsigned char GetRow() const { return fRow; }
	GPUhd() unsigned char GetFlags() const { return fFlags; }
	GPUhd() unsigned short GetAmp() const { return fAmp; }

  private:
	unsigned int fId;     // Id ( slice, patch, cluster )
	unsigned char fRow;   // row
	unsigned char fFlags; //flags
	unsigned short fAmp;  //amplitude
	float fX;             // coordinates
	float fY;             // coordinates
	float fZ;             // coordinates

#ifdef GMPropagatePadRowTime
  public:
	float fPad;
	float fTime;
#endif
};

#endif
