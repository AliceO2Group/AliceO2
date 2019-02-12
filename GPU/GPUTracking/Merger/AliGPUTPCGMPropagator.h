//-*- Mode: C++ -*-
// $Id: AliGPUTPCGMPropagator.h 39008 2010-02-18 17:33:32Z sgorbuno $
// ************************************************************************
// This file is property of and copyright by the ALICE HLT Project        *
// ALICE Experiment at CERN, All rights reserved.                         *
// See cxx source for full Copyright notice                               *
//                                                                        *
//*************************************************************************

#ifndef AliGPUTPCGMPropagator_H
#define AliGPUTPCGMPropagator_H

#include "AliGPUCommonDef.h"
#include "AliGPUTPCGMOfflineStatisticalErrors.h"
#include "AliGPUTPCGMPhysicalTrackModel.h"
#include "AliGPUTPCGMPolynomialField.h"
#include "AliGPUCommonMath.h"

class AliGPUTPCGMTrackParam;
class AliGPUParam;

/**
 * @class AliGPUTPCGMPropagator
 *
 */

class AliGPUTPCGMPropagator
{
  public:
	/// Enumeration of field regions
	enum FieldRegion
	{
		TPC = 0, ///< TPC
		TRD = 1, ///< outer TPC -> outer TRD
	};

	GPUd() AliGPUTPCGMPropagator();

	struct MaterialCorrection
	{
		GPUd() MaterialCorrection() : fRadLen(29.532f), fRho(1.025e-3f), fRhoOverRadLen(fRho / fRadLen),
		                              fDLMax(0.f), fEP2(0.f), fSigmadE2(0.f), fK22(0.f), fK33(0.f), fK43(0.f), fK44(0.f) {}

		float fRadLen, fRho, fRhoOverRadLen,
		    fDLMax, fEP2, fSigmadE2, fK22, fK33, fK43, fK44; // precalculated values for MS and EnergyLoss correction
	};

	GPUd() void SetMaterial(float radLen, float rho);

	GPUd() void SetPolynomialField(const AliGPUTPCGMPolynomialField *field) { fField = field; }

	GPUd() void SelectFieldRegion(FieldRegion region) { fFieldRegion = region; }

	GPUd() void SetFitInProjections(bool Flag) { fFitInProjections = Flag; }
	GPUd() void SetToyMCEventsFlag(bool Flag) { fToyMCEvents = Flag; }
	GPUd() void SetSpecialErrors(bool Flag) { fSpecialErrors = Flag; }

	GPUd() void SetMaxSinPhi(float maxSinPhi) { fMaxSinPhi = maxSinPhi; }

	GPUd() void SetTrack(AliGPUTPCGMTrackParam *track, float Alpha);
	GPUd() void ResetT0()
	{
		if (!fT) return;
		fT0.Set(*fT);
	}

	GPUd() int RotateToAlpha(float newAlpha);

	GPUd() int PropagateToXAlpha(float posX, float posAlpha, bool inFlyDirection);

	//  GPUd() int PropagateToXAlphaBz( float posX, float posAlpha, bool inFlyDirection );

	GPUd() int Update(float posY, float posZ, int iRow, const AliGPUParam &param, short clusterState, bool rejectChi2, bool refit);
	GPUd() int Update(float posY, float posZ, short clusterState, bool rejectChi2, float err2Y, float err2Z);
	GPUd() float PredictChi2(float posY, float posZ, int iRow, const AliGPUParam &param, short clusterState) const;
	GPUd() float PredictChi2(float posY, float posZ, float err2Y, float err2Z) const;
	GPUd() int RejectCluster(float chiY, float chiZ, unsigned char clusterState)
	{
		if (chiY > 9.f || chiZ > 9.f) return 2;
		if ((chiY > 6.25f || chiZ > 6.25f) && (clusterState & (AliGPUTPCGMMergedTrackHit::flagSplit | AliGPUTPCGMMergedTrackHit::flagShared))) return 2;
		if ((chiY > 1.f || chiZ > 6.25f) && (clusterState & (AliGPUTPCGMMergedTrackHit::flagEdge | AliGPUTPCGMMergedTrackHit::flagSingle))) return 2;
		return 0;
	}

	GPUd() float GetBz(float Alpha, float X, float Y, float Z) const;
	GPUd() void GetBxByBz(float Alpha, float X, float Y, float Z, float B[3]) const;

	GPUd() void GetErr2(float &err2Y, float &err2Z, const AliGPUParam &param, float posZ, int iRow, short clusterState) const;

	GPUd() float GetAlpha() const { return fAlpha; }
	GPUd() float GetQPt0() const { return fT0.GetQPt(); }
	GPUd() float GetSinPhi0() const { return fT0.GetSinPhi(); }
	GPUd() float GetCosPhi0() const { return fT0.GetCosPhi(); }
	GPUd() void Mirror(bool inFlyDirection);
	GPUd() void Rotate180();
	GPUd() void ChangeDirection();
	GPUd() float GetMirroredYModel() const;
	GPUd() float GetMirroredYTrack() const;
	GPUd() int GetPropagatedYZ(float x, float &projY, float &projZ);
	GPUd() bool GetFitInProjections() const { return fFitInProjections; }

	GPUd() AliGPUTPCGMPhysicalTrackModel &Model() { return fT0; }
	GPUd() void CalculateMaterialCorrection();
	GPUd() void SetStatErrorCurCluster(AliGPUTPCGMMergedTrackHit *c) { fStatErrors.SetCurCluster(c); }

  private:
	GPUd() static float ApproximateBetheBloch(float beta2);

	const AliGPUTPCGMPolynomialField *fField;
	FieldRegion fFieldRegion;

	AliGPUTPCGMTrackParam *fT;
	float fAlpha; // rotation angle of the track coordinate system
	AliGPUTPCGMPhysicalTrackModel fT0;
	MaterialCorrection fMaterial;
	bool fSpecialErrors;
	bool fFitInProjections; // fit (Y,SinPhi,QPt) and (Z,DzDs) paramteres separatelly
	bool fToyMCEvents;      // events are simulated with simple home-made simulation
	float fMaxSinPhi;

	AliGPUTPCGMOfflineStatisticalErrors fStatErrors;
};

GPUd() inline AliGPUTPCGMPropagator::AliGPUTPCGMPropagator()
    : fField(0), fFieldRegion(TPC), fT(0), fAlpha(0), fT0(), fMaterial(),
      fSpecialErrors(0), fFitInProjections(1), fToyMCEvents(0), fMaxSinPhi(GPUCA_MAX_SIN_PHI), fStatErrors()
{
}

GPUd() inline void AliGPUTPCGMPropagator::SetMaterial(float radLen, float rho)
{
	fMaterial.fRho = rho;
	fMaterial.fRadLen = radLen;
	fMaterial.fRhoOverRadLen = (radLen > 1.e-4f) ? rho / radLen : 0.f;
	CalculateMaterialCorrection();
}

GPUd() inline void AliGPUTPCGMPropagator::SetTrack(AliGPUTPCGMTrackParam *track, float Alpha)
{
	fT = track;
	if (!fT) return;
	fT0.Set(*fT);
	fAlpha = Alpha;
	CalculateMaterialCorrection();
}

GPUd() inline float AliGPUTPCGMPropagator::GetMirroredYModel() const
{
	float Bz = GetBz(fAlpha, fT0.GetX(), fT0.GetY(), fT0.GetZ());
	return fT0.GetMirroredY(Bz);
}

GPUd() inline float AliGPUTPCGMPropagator::GetMirroredYTrack() const
{
	if (!fT) return -1.E10f;
	float Bz = GetBz(fAlpha, fT->GetX(), fT->GetY(), fT->GetZ());
	return fT->GetMirroredY(Bz);
}

#endif
