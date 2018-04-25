#ifndef ALIHLTTPCGMOFFLINESTATISTICALERRORS
#define ALIHLTTPCGMOFFLINESTATISTICALERRORS

class AliHLTTPCGMMergedTrackHit;

#if defined(GPUseStatError)
#include "AliTPCcalibDB.h"
#include "AliTPCclusterMI.h"
#include "AliTPCtracker.h"
#include "AliTPCTransform.h"
#include "AliHLTTPCGMMergedTrackHit.h"
#include "AliTPCReconstructor.h"

struct AliHLTTPCGMOfflineStatisticalErrors
{
	void SetCurCluster(AliHLTTPCGMMergedTrackHit* c) {fCurCluster = c;}

	void GetOfflineStatisticalErrors(float& err2Y, float& err2Z, float sinPhi, float dzds, unsigned char clusterState)
	{
		float snp2 = sinPhi * sinPhi;
		if (snp2>1.-1e-6) snp2 = 1.-1e-6;
		float tgp2 = snp2/(1.f-snp2);
		float tgp  = sqrt(tgp2);
		double serry2=0,serrz2=0;
		AliTPCclusterMI cl;
		cl.SetRow(fCurCluster->fRow);
		cl.SetPad(fCurCluster->fPad);
		cl.SetTimeBin(fCurCluster->fTime);
		int type = 0;
		if (clusterState & AliHLTTPCGMMergedTrackHit::flagSplit) type = 50;
		if (clusterState & AliHLTTPCGMMergedTrackHit::flagEdge) type = -type - 3;
		cl.SetType(type);
		cl.SetSigmaY2(0.5);
		cl.SetSigmaZ2(0.5);
		cl.SetQ(fCurCluster->fAmp);
		cl.SetMax(25);
		cl.SetX(fCurCluster->fX);
		cl.SetY(fCurCluster->fY);
		cl.SetZ(fCurCluster->fZ);
		cl.SetDetector(fCurCluster->fSlice);
		if (fCurCluster->fRow >= 63)
		{
		  cl.SetRow(fCurCluster->fRow - 63);
		  cl.SetDetector(fCurCluster->fSlice + 36);
		}
		static AliTPCtracker trk;
		/*AliTPCRecoParam *par = const_cast<AliTPCRecoParam*>(AliTPCReconstructor::GetRecoParam()), *par2;
		AliTPCReconstructor rec;
		if (par == NULL)
		{
			par2 = new AliTPCRecoParam;
			par2->SetUseSectorAlignment(false);
			rec.SetRecoParam(par2);
		}*/
		//This needs AliTPCRecoParam::GetUseSectorAlignment(), so we should make sure that the RecoParam exists!
		//This is not true during HLT simulation by definition, so the above code should fix it, but it leads to non-understandable bug later on in TPC transformation.
		//Anyway, this is only a debugging class.
		//So in order to make this work, please temporarily outcomment any use of TPCRecoParam in AliTPCTracker::Transform (it is not needed here anyway...)
		trk.Transform(&cl); 
		/*if (par == NULL)
		{
			delete par2;
			rec.SetRecoParam(NULL);
		}*/

		AliTPCcalibDB::Instance()->GetTransform()->ErrY2Z2Syst(&cl, tgp, dzds, serry2, serrz2);

		//printf("TEST Sector %d Row %d: X %f %f Y %f %f Z %f %f - Err Y %f + %f, Z %f + %f\n", fCurCluster->fSlice, fCurCluster->fRow, fCurCluster->fX, cl.GetX(), fCurCluster->fY, cl.GetY(), fCurCluster->fZ, cl.GetZ(), err2Y, serry2, err2Z, serrz2);

		err2Y += serry2;
		err2Z += serrz2;
	}
	
	AliHLTTPCGMMergedTrackHit* fCurCluster;
};
#else
struct AliHLTTPCGMOfflineStatisticalErrors
{
	GPUd() void SetCurCluster(AliHLTTPCGMMergedTrackHit* c) {}
	GPUd() void GetOfflineStatisticalErrors(float& err2Y, float& err2Z, float sinPhi, float dzds, unsigned char clusterState) {}
};
#endif

#endif
