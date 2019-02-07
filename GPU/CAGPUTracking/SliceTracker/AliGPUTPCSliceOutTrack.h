//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCCASLICEOUTTRACK_H
#define ALIHLTTPCCASLICEOUTTRACK_H

#include "AliGPUTPCBaseTrackParam.h"
#include "AliGPUTPCSliceOutCluster.h"

/**
 * @class AliGPUTPCSliceOutTrack
 * AliGPUTPCSliceOutTrack class is used to store TPC tracks,
 * which are reconstructed by the TPCCATracker slice tracker.
 *
 * The class contains:
 * - fitted track parameters at its first row, the covariance matrix, \Chi^2, NDF (number of degrees of freedom )
 * - n of clusters assigned to the track
 * - clusters in corresponding cluster arrays
 *
 * The class is used to transport the data between AliGPUTPCTracker{Component} and AliGPUTPCGBMerger{Component}
 *
 */
class AliGPUTPCSliceOutTrack
{
  public:
	GPUhd() int NClusters() const { return fNClusters; }
	GPUhd() const AliGPUTPCBaseTrackParam &Param() const { return fParam; }
	GPUhd() const AliGPUTPCSliceOutCluster &Cluster(int i) const { return fClusters[i]; }
	GPUhd() const AliGPUTPCSliceOutCluster *Clusters() const { return fClusters; }

	GPUhd() void SetNClusters(int v) { fNClusters = v; }
	GPUhd() void SetParam(const AliGPUTPCBaseTrackParam &v) { fParam = v; }
	GPUhd() void SetCluster(int i, const AliGPUTPCSliceOutCluster &v) { fClusters[i] = v; }

	GPUhd() static int GetSize(int nClust) { return sizeof(AliGPUTPCSliceOutTrack) + nClust * sizeof(AliGPUTPCSliceOutCluster); }

	GPUhd() int LocalTrackId() const { return fLocalTrackId; }
	GPUhd() void SetLocalTrackId(int v) { fLocalTrackId = v; }

	GPUhd() AliGPUTPCSliceOutTrack *NextTrack()
	{
		return (AliGPUTPCSliceOutTrack *) (((char *) this) + GetSize(fNClusters));
	}

	GPUhd() const AliGPUTPCSliceOutTrack *GetNextTrack() const
	{
		return (AliGPUTPCSliceOutTrack *) (((char *) this) + GetSize(fNClusters));
	}

  private:
	AliGPUTPCBaseTrackParam fParam; //* fitted track parameters at its innermost cluster
	int fNClusters;                   //* number of track clusters
	int fLocalTrackId;                //See AliHLTPCCATrack.h
	AliGPUTPCSliceOutCluster fClusters[0]; //* track clusters
};

#endif
