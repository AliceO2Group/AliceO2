// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCSliceOutTrack.h
/// \author Sergey Gorbunov, David Rohr

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
