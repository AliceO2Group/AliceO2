#ifndef ALINEUTRALTRACKPARAM_H
#define ALINEUTRALTRACKPARAM_H
/* Copyright(c) 1998-2009, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/*****************************************************************************
 *              "Neutral" track parametrisation class                        *
 * At the moment we use a standard AliExternalTrackParam with 0 curvature.   *
 *                                                                           *
 *        Origin: A.Dainese, I.Belikov                                       *
 *****************************************************************************/

#include "AliExternalTrackParam.h"

class AliNeutralTrackParam: public AliExternalTrackParam {
 public:
  AliNeutralTrackParam();
  AliNeutralTrackParam(const AliNeutralTrackParam &);
  AliNeutralTrackParam& operator=(const AliNeutralTrackParam & trkPar);
  AliNeutralTrackParam(Double_t x, Double_t alpha, 
			const Double_t param[5], const Double_t covar[15]);
  AliNeutralTrackParam(const AliVTrack *vTrack);
  AliNeutralTrackParam(Double_t xyz[3],Double_t pxpypz[3],
			Double_t cv[21],Short_t sign);
  virtual ~AliNeutralTrackParam(){}

  virtual Short_t  Charge() const { return 0; }
  virtual Double_t GetC(Double_t /*b*/) const { return 0.; }

 private:

  ClassDef(AliNeutralTrackParam, 1) // track with zero charge
};

#endif
