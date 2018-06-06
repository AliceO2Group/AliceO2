// $Id$
//* This file is property of and copyright by the ALICE HLT Project        *
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTRDINTERFACES_H
#define ALIHLTTRDINTERFACES_H

#include "AliExternalTrackParam.h"
#include "AliTrackerBase.h"

/**
 * @class this is an interface header for making the TRD tracking portable between O2, AliRoot, and HLT standalone framework
 */

template <typename T> class trackInterface;
template <> class trackInterface<AliExternalTrackParam> : public AliExternalTrackParam
{
  typedef double My_Float;
  typedef trackInterface<AliExternalTrackParam> interface;

  public:
    trackInterface<AliExternalTrackParam>() : AliExternalTrackParam() {};
    trackInterface<AliExternalTrackParam>(const trackInterface<AliExternalTrackParam> &param) : AliExternalTrackParam(param) {};
    trackInterface<AliExternalTrackParam>(const AliExternalTrackParam &param) : AliExternalTrackParam(param) {};

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

    // parameter manipulation
    bool update(const My_Float p[2], const My_Float cov[3])                  { return Update(p, cov); }
    float getPredictedChi2(const My_Float p[2], const My_Float cov[3]) const { return GetPredictedChi2(p, cov); }
    bool rotate(float alpha)                                                 { return Rotate(alpha); }

    void set(float x, float alpha, const float param[5], const float cov[15])  { Set(x, alpha, param, cov); }

    typedef AliExternalTrackParam baseClass;
};





template <typename T> class propagatorInterface;
template <> class propagatorInterface<AliTrackerBase> : public AliTrackerBase
{
  public:
    bool PropagateToX(AliExternalTrackParam *trk, float x, float maxSnp, float maxStep) {
      return PropagateTrackToBxByBz(trk, x, 0.13957, maxStep, false, maxSnp);
    }
};
#endif
