// $Id$
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIGPUTRDINTERFACES_H
#define ALIGPUTRDINTERFACES_H

/**
 * @class this is an interface header for making the TRD tracking portable between O2, AliRoot, and HLT standalone framework
 */

#include "AliTPCCommonDef.h"
#include "AliTPCCommonMath.h"
#include "AliGPUTPCGMMergedTrack.h"
#include "AliGPUTPCGMTrackParam.h"
#include "AliGPUTRDDef.h"
template <typename T> class trackInterface;
template <typename T> class propagatorInterface;

#ifdef GPUCA_ALIROOT_LIB //Interface for AliRoot, build only with AliRoot
#include "AliExternalTrackParam.h"
#include "AliHLTExternalTrackParam.h"
#include "AliTrackerBase.h"

template <> class trackInterface<AliExternalTrackParam> : public AliExternalTrackParam
{

  public:
    trackInterface<AliExternalTrackParam> () : AliExternalTrackParam() {};
    trackInterface<AliExternalTrackParam> (const trackInterface<AliExternalTrackParam> &param) :
      AliExternalTrackParam(param) {};
    trackInterface<AliExternalTrackParam> (const AliExternalTrackParam &param) CON_DELETE;
    trackInterface<AliExternalTrackParam> (const AliHLTExternalTrackParam &param) :
      AliExternalTrackParam()
    {
      float paramTmp[5] = { param.fY, param.fZ, param.fSinPhi, param.fTgl, param.fq1Pt };
      Set( param.fX, param.fAlpha, paramTmp, param.fC );
    }
    trackInterface<AliExternalTrackParam>(const AliGPUTPCGMMergedTrack &trk) :
      AliExternalTrackParam()
    {
      Set( trk.GetParam().GetX(), trk.GetAlpha(), trk.GetParam().GetPar(), trk.GetParam().GetCov() );
    }
    trackInterface<AliExternalTrackParam>(const AliGPUTPCGMTrackParam::AliGPUTPCOuterParam &param) :
      AliExternalTrackParam()
    {
      Set( param.fX, param.fAlpha, param.fP, param.fC );
    }

    // parameter + covariance
    float getX()       const { return GetX(); }
    float getAlpha()   const { return GetAlpha(); }
    float getY()       const { return GetY(); }
    float getZ()       const { return GetZ(); }
    float getSnp()     const { return GetSnp(); }
    float getTgl()     const { return GetTgl(); }
    float getQ2Pt()    const { return GetSigned1Pt(); }
    float getEta()     const { return Eta(); }
    float getPt()      const { return Pt(); }
    float getSigmaY2() const { return GetSigmaY2(); }
    float getSigmaZ2() const { return GetSigmaZ2(); }

    const My_Float *getCov() const { return GetCovariance(); }
    bool CheckNumericalQuality() const { return true; }

    // parameter manipulation
    bool update(const My_Float p[2], const My_Float cov[3])                  { return Update(p, cov); }
    float getPredictedChi2(const My_Float p[2], const My_Float cov[3]) const { return GetPredictedChi2(p, cov); }
    bool rotate(float alpha)                                                 { return Rotate(alpha); }

    void set(float x, float alpha, const float param[5], const float cov[15])  { Set(x, alpha, param, cov); }

    typedef AliExternalTrackParam baseClass;
};

template <> class propagatorInterface<AliTrackerBase> : public AliTrackerBase
{

  public:
    propagatorInterface<AliTrackerBase> (const void* = nullptr) : AliTrackerBase(), fParam(nullptr) {};
    propagatorInterface<AliTrackerBase> (const propagatorInterface<AliTrackerBase>&) CON_DELETE;
    propagatorInterface<AliTrackerBase>& operator=(const propagatorInterface<AliTrackerBase>&) CON_DELETE;

    bool PropagateToX(float x, float maxSnp, float maxStep) {
      return PropagateTrackToBxByBz(fParam, x, 0.13957, maxStep, false, maxSnp);
    }

    void setTrack(trackInterface<AliExternalTrackParam> *trk) { fParam = trk; }

    float getAlpha() { return (fParam) ? fParam->GetAlpha() : 99999.f; }
    bool update(const My_Float p[2], const My_Float cov[3]) { return (fParam) ? fParam->update(p, cov) : false; }
    float getPredictedChi2(const My_Float p[2], const My_Float cov[3]) { return (fParam) ? fParam->getPredictedChi2(p, cov) : 99999.f; }
    bool rotate(float alpha) { return (fParam) ? fParam->rotate(alpha) : false; }

    trackInterface<AliExternalTrackParam> *fParam;
};

#endif

#ifdef GPUCA_O2_LIB //Interface for O2, build only with O2
//TODO: Implement!
#endif

#include "AliGPUTPCGMPropagator.h"
#include "AliGPUTPCGMMerger.h"
#include "AliGPUCAParam.h"
#include "AliGPUTPCDef.h"

template <> class trackInterface<AliGPUTPCGMTrackParam> : public AliGPUTPCGMTrackParam
{
  public:
    GPUd() trackInterface<AliGPUTPCGMTrackParam>() : AliGPUTPCGMTrackParam(), fAlpha(0.f) {};
    GPUd() trackInterface<AliGPUTPCGMTrackParam>(const AliGPUTPCGMTrackParam &param) CON_DELETE;
    GPUd() trackInterface<AliGPUTPCGMTrackParam>(const AliGPUTPCGMMergedTrack &trk) :
      AliGPUTPCGMTrackParam(),
      fAlpha(trk.GetAlpha())
    {
      SetX(trk.GetParam().GetX());
      SetPar(0, trk.GetParam().GetY());
      SetPar(1, trk.GetParam().GetZ());
      SetPar(2, trk.GetParam().GetSinPhi());
      SetPar(3, trk.GetParam().GetDzDs());
      SetPar(4, trk.GetParam().GetQPt());
      for (int i=0; i<15; i++) {
        SetCov(i, trk.GetParam().GetCov(i));
      }
    };
    GPUd() trackInterface<AliGPUTPCGMTrackParam>(const AliGPUTPCGMTrackParam::AliGPUTPCOuterParam &param) :
      AliGPUTPCGMTrackParam(),
      fAlpha(param.fAlpha)
    {
      SetX(param.fX);
      for (int i=0; i<5; i++) {
        SetPar(i, param.fP[i]);
      }
      for (int i=0; i<15; i++) {
        SetCov(i, param.fC[i]);
      }
    };
    GPUd() trackInterface<AliGPUTPCGMTrackParam>(const trackInterface<AliGPUTPCGMTrackParam> &param) :
      AliGPUTPCGMTrackParam(),
      fAlpha(param.fAlpha)
    {
      SetX(param.getX());
      for (int i=0; i<5; i++) {
        SetPar(i, param.GetPar(i));
      }
      for (int j=0; j<15; j++) {
        SetCov(j, param.GetCov(j));
      }
    }
#ifdef GPUCA_ALIROOT_LIB
    trackInterface<AliGPUTPCGMTrackParam>(const AliHLTExternalTrackParam &param) :
      AliGPUTPCGMTrackParam(),
      fAlpha(param.fAlpha)
    {
      SetX(param.fX);
      SetPar(0, param.fY);
      SetPar(1, param.fZ);
      SetPar(2, param.fSinPhi);
      SetPar(3, param.fTgl);
      SetPar(4, param.fq1Pt);
      for (int i=0; i<15; i++) {
        SetCov(i, param.fC[i]);
      }
    };
#endif

    GPUd() float getX()       const { return GetX(); }
    GPUd() float getAlpha()   const { return fAlpha; }
    GPUd() float getY()       const { return GetY(); }
    GPUd() float getZ()       const { return GetZ(); }
    GPUd() float getSnp()     const { return GetSinPhi(); }
    GPUd() float getTgl()     const { return GetDzDs(); }
    GPUd() float getQ2Pt()    const { return GetQPt(); }
    GPUd() float getEta()     const { return -CAMath::Log( CAMath::Tan( 0.5f * (0.5f * M_PI - CAMath::ATan(getTgl())) ) ); }
    GPUd() float getPt()      const { return CAMath::Abs(getQ2Pt()) > 0 ? CAMath::Abs(1.f/getQ2Pt()) : 99999.f; }
    GPUd() float getSigmaY2() const { return GetErr2Y(); }
    GPUd() float getSigmaZ2() const { return GetErr2Z(); }

    GPUd() const float *getCov() const { return GetCov(); }

    GPUd() void setAlpha(float alpha) { fAlpha = alpha; }
    GPUd() void set(float x, float alpha, const float param[5], const float cov[15]) {
      SetX(x);
      for (int i=0; i<5; i++) {
        SetPar(i, param[i]);
      }
      for (int j=0; j<15; j++) {
        SetCov(j, cov[j]);
      }
      setAlpha(alpha);
    }

    typedef AliGPUTPCGMTrackParam baseClass;

  private:
    float fAlpha;
};

template <> class propagatorInterface<AliGPUTPCGMPropagator> : public AliGPUTPCGMPropagator
{
  public:
    GPUd() propagatorInterface<AliGPUTPCGMPropagator>(const AliGPUTPCGMMerger *pMerger) : AliGPUTPCGMPropagator(), fTrack(nullptr) {
      constexpr float kRho = 1.025e-3f;
      constexpr float kRadLen = 29.532f;
      this->SetMaterial( kRadLen, kRho );
      this->SetPolynomialField( pMerger->pField() );
      this->SetMaxSinPhi( GPUCA_MAX_SIN_PHI );
      this->SetToyMCEventsFlag(0);
      this->SetFitInProjections(0);
      this->SelectFieldRegion(AliGPUTPCGMPropagator::TRD);
    };
    propagatorInterface<AliGPUTPCGMPropagator>(const propagatorInterface<AliGPUTPCGMPropagator>&) CON_DELETE;
    propagatorInterface<AliGPUTPCGMPropagator>& operator=(const propagatorInterface<AliGPUTPCGMPropagator>&) CON_DELETE;
    GPUd() void setTrack(trackInterface<AliGPUTPCGMTrackParam> *trk) { SetTrack(trk, trk->getAlpha()); fTrack = trk;}
    GPUd() bool PropagateToX( float x, float maxSnp, float maxStep ) {
      bool ok = PropagateToXAlpha( x, GetAlpha(), true ) == 0 ? true : false;
      ok = fTrack->CheckNumericalQuality();
      return ok;
    }
    GPUd() bool rotate(float alpha) {
      if (RotateToAlpha(alpha) == 0) {
        fTrack->setAlpha(alpha);
        return fTrack->CheckNumericalQuality();
      }
      return false;
    }
    GPUd() bool update(const My_Float p[2], const My_Float cov[3]) {
      // TODO sigma_yz not taken into account yet, is not zero due to pad tilting!
      return Update(p[0], p[1], 0, false, cov[0], cov[2]) == 0 ? true : false;
    }
    GPUd() float getAlpha() { return GetAlpha(); }
    // TODO sigma_yz not taken into account yet, is not zero due to pad tilting!
    GPUd() float getPredictedChi2(const My_Float p[2], const My_Float cov[3]) const { return PredictChi2( p[0], p[1], cov[0], cov[2]); }

    trackInterface<AliGPUTPCGMTrackParam> *fTrack;
};

#endif
