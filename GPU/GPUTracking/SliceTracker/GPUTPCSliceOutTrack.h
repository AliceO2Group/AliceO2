//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef GPUTPCSLICEOUTTRACK_H
#define GPUTPCSLICEOUTTRACK_H

#include "GPUTPCBaseTrackParam.h"
#include "GPUTPCSliceOutCluster.h"

/**
 * @class GPUTPCSliceOutTrack
 * GPUTPCSliceOutTrack class is used to store TPC tracks,
 * which are reconstructed by the TPCCATracker slice tracker.
 *
 * The class contains:
 * - fitted track parameters at its first row, the covariance matrix, \Chi^2, NDF (number of degrees of freedom )
 * - n of clusters assigned to the track
 * - clusters in corresponding cluster arrays
 *
 * The class is used to transport the data between GPUTPCTracker{Component} and GPUTPCGBMerger{Component}
 *
 */
class GPUTPCSliceOutTrack
{
  public:
	GPUhd() int NClusters() const { return fNClusters; }
	GPUhd() const GPUTPCBaseTrackParam &Param() const { return fParam; }
	GPUhd() const GPUTPCSliceOutCluster &Cluster(int i) const { return fClusters[i]; }
	GPUhd() const GPUTPCSliceOutCluster *Clusters() const { return fClusters; }

	GPUhd() void SetNClusters(int v) { fNClusters = v; }
	GPUhd() void SetParam(const GPUTPCBaseTrackParam &v) { fParam = v; }
	GPUhd() void SetCluster(int i, const GPUTPCSliceOutCluster &v) { fClusters[i] = v; }

	GPUhd() static int GetSize(int nClust) { return sizeof(GPUTPCSliceOutTrack) + nClust * sizeof(GPUTPCSliceOutCluster); }

	GPUhd() int LocalTrackId() const { return fLocalTrackId; }
	GPUhd() void SetLocalTrackId(int v) { fLocalTrackId = v; }

	GPUhd() GPUTPCSliceOutTrack *NextTrack()
	{
		return (GPUTPCSliceOutTrack *) (((char *) this) + GetSize(fNClusters));
	}

	GPUhd() const GPUTPCSliceOutTrack *GetNextTrack() const
	{
		return (GPUTPCSliceOutTrack *) (((char *) this) + GetSize(fNClusters));
	}

  private:
	GPUTPCBaseTrackParam fParam; //* fitted track parameters at its innermost cluster
	int fNClusters;                   //* number of track clusters
	int fLocalTrackId;                //See AliHLTPCCATrack.h
	GPUTPCSliceOutCluster fClusters[0]; //* track clusters
};

#endif
