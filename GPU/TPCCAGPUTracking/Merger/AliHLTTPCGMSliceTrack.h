//-*- Mode: C++ -*-
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef ALIHLTTPCGMSLICETRACK_H
#define ALIHLTTPCGMSLICETRACK_H

#include "AliHLTTPCCASliceOutTrack.h"
#include "AliHLTTPCGMTrackParam.h"
#include "AliTPCCommonMath.h"
#include <algorithm>

/**
 * @class AliHLTTPCGMSliceTrack
 *
 * The class describes TPC slice tracks used in AliHLTTPCGMMerger
 */
class AliHLTTPCGMSliceTrack
{

  public:
	float Alpha() const { return fAlpha; }
	char Slice() const { return (char) fSlice; }
	char CSide() const { return fSlice >= 18; }
	int NClusters() const { return fNClusters; }
	int PrevNeighbour() const { return fNeighbour[0]; }
	int NextNeighbour() const { return fNeighbour[1]; }
	int Neighbour(int i) const { return fNeighbour[i]; }
	int PrevSegmentNeighbour() const { return fSegmentNeighbour[0]; }
	int NextSegmentNeighbour() const { return fSegmentNeighbour[1]; }
	int SegmentNeighbour(int i) const { return fSegmentNeighbour[i]; }
	const AliHLTTPCCASliceOutTrack *OrigTrack() const { return fOrigTrack; }
	float X() const { return fX; }
	float Y() const { return fY; }
	float Z() const { return fZ; }
	float SinPhi() const { return fSinPhi; }
	float CosPhi() const { return fCosPhi; }
	float SecPhi() const { return fSecPhi; }
	float DzDs() const { return fDzDs; }
	float QPt() const { return fQPt; }
	float ZOffset() const { return fZOffset; }
	float Leg() const { return fLeg; }

	int LocalTrackId() const { return fLocalTrackId; }
	void SetLocalTrackId(int v) { fLocalTrackId = v; }
	int GlobalTrackId(int n) const { return fGlobalTrackIds[n]; }
	void SetGlobalTrackId(int n, int v) { fGlobalTrackIds[n] = v; }

	float MaxClusterZ() { return std::max(fOrigTrack->Clusters()->GetZ(), (fOrigTrack->Clusters() + fOrigTrack->NClusters() - 1)->GetZ()); }
	float MinClusterZ() { return std::min(fOrigTrack->Clusters()->GetZ(), (fOrigTrack->Clusters() + fOrigTrack->NClusters() - 1)->GetZ()); }

	void Set(const AliHLTTPCCASliceOutTrack *sliceTr, float alpha, int slice)
	{
		const AliHLTTPCCABaseTrackParam &t = sliceTr->Param();
		fOrigTrack = sliceTr;
		fX = t.GetX();
		fY = t.GetY();
		fZ = t.GetZ();
		fDzDs = t.GetDzDs();
		fSinPhi = t.GetSinPhi();
		fQPt = t.GetQPt();
		fCosPhi = sqrt(1.f - fSinPhi * fSinPhi);
		fSecPhi = 1.f / fCosPhi;
		fAlpha = alpha;
		fSlice = slice;
		fZOffset = t.GetZOffset();
		fNClusters = sliceTr->NClusters();
	}

	void SetGlobalSectorTrackCov()
	{
		fC0 = 1;
		fC2 = 1;
		fC3 = 0;
		fC5 = 1;
		fC7 = 0;
		fC9 = 1;
		fC10 = 0;
		fC12 = 0;
		fC14 = 10;
	}

	void SetNClusters(int v) { fNClusters = v; }
	void SetPrevNeighbour(int v) { fNeighbour[0] = v; }
	void SetNextNeighbour(int v) { fNeighbour[1] = v; }
	void SetNeighbor(int v, int i) { fNeighbour[i] = v; }
	void SetPrevSegmentNeighbour(int v) { fSegmentNeighbour[0] = v; }
	void SetNextSegmentNeighbour(int v) { fSegmentNeighbour[1] = v; }
	void SetLeg(unsigned char v) { fLeg = v; }

	void CopyParamFrom(const AliHLTTPCGMSliceTrack &t)
	{
		fX = t.fX;
		fY = t.fY;
		fZ = t.fZ;
		fSinPhi = t.fSinPhi;
		fDzDs = t.fDzDs;
		fQPt = t.fQPt;
		fCosPhi = t.fCosPhi, fSecPhi = t.fSecPhi;
		fAlpha = t.fAlpha;
	}

	bool FilterErrors(const AliGPUCAParam &param, float maxSinPhi = HLTCA_MAX_SIN_PHI, float sinPhiMargin = 0.f);

	bool TransportToX(float x, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi, bool doCov = true) const;

	bool TransportToXAlpha(float x, float sinAlpha, float cosAlpha, float Bz, AliHLTTPCGMBorderTrack &b, float maxSinPhi) const;

  private:
	const AliHLTTPCCASliceOutTrack *fOrigTrack;               // pointer to original slice track
	float fX, fY, fZ, fSinPhi, fDzDs, fQPt, fCosPhi, fSecPhi; // parameters
	float fZOffset;
	float fC0, fC2, fC3, fC5, fC7, fC9, fC10, fC12, fC14; // covariances
	float fAlpha;                                         // alpha angle
	int fSlice;                                           // slice of this track segment
	int fNClusters;                                       // N clusters
	int fNeighbour[2];                                    //
	int fSegmentNeighbour[2];                             //
	int fLocalTrackId;                                    // Corrected local track id in terms of GMSliceTracks array
	int fGlobalTrackIds[2];                               // IDs of associated global tracks
	unsigned char fLeg;                                   //Leg of this track segment
};

#endif
