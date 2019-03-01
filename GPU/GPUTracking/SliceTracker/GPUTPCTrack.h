//-*- Mode: C++ -*-
// @(#) $Id$
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef GPUTPCTRACK_H
#define GPUTPCTRACK_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCDef.h"

/**
 * @class GPUTPCtrack
 *
 * The class describes the [partially] reconstructed TPC track [candidate].
 * The class is dedicated for internal use by the GPUTPCTracker algorithm.
 * The track parameters at both ends are stored separately in the GPUTPCEndPoint class
 */
MEM_CLASS_PRE()
class GPUTPCTrack
{
  public:
#if !defined(GPUCA_GPUCODE)
	GPUTPCTrack() : fAlive(0), fFirstHitID(0), fNHits(0), fLocalTrackId(-1), fParam()
	{
	}
	~GPUTPCTrack() {}
#endif //!GPUCA_GPUCODE

	GPUhd() char Alive() const
	{
		return fAlive;
	}
	GPUhd() int NHits() const { return fNHits; }
	GPUhd() int LocalTrackId() const { return fLocalTrackId; }
	GPUhd() int FirstHitID() const { return fFirstHitID; }
	GPUhd() MakeType(const MEM_LG(GPUTPCBaseTrackParam) &) Param() const { return fParam; }

	GPUhd() void SetAlive(bool v) { fAlive = v; }
	GPUhd() void SetNHits(int v) { fNHits = v; }
	GPUhd() void SetLocalTrackId(int v) { fLocalTrackId = v; }
	GPUhd() void SetFirstHitID(int v) { fFirstHitID = v; }

	MEM_TEMPLATE()
	GPUhd() void SetParam(const MEM_TYPE(GPUTPCBaseTrackParam) & v) { fParam = v; }

  private:
	char fAlive;       // flag for mark tracks used by the track merger
	int fFirstHitID;   // index of the first track cell in the track->cell pointer array
	int fNHits;        // number of track cells
	int fLocalTrackId; //Id of local track this global track belongs to, index of this track itself if it is a local track
	MEM_LG(GPUTPCBaseTrackParam)
	fParam; // track parameters

  private:
	//void Dummy(); // to make rulechecker happy by having something in .cxx file

	//ClassDef(GPUTPCTrack,1)
};

#endif //GPUTPCTRACK_H
