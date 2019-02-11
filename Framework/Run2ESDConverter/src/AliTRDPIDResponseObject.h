/**************************************************************************
* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
*                                                                        *
* Author: The ALICE Off-line Project.                                    *
* Contributors are mentioned in the code where appropriate.              *
*                                                                        *
* Permission to use, copy, modify and distribute this software and its   *
* documentation strictly for non-commercial purposes is hereby granted   *
* without fee, provided that the above copyright notice appears in all   *
* copies and that both the copyright notice and this permission notice   *
* appear in the supporting documentation. The authors make no claims     *
* about the suitability of this software for any purpose. It is          *
* provided "as is" without express or implied warranty.                  *
**************************************************************************/
//
// Container for TRD PID Response Objects stored in the OADB
//
#ifndef ALITRDPIDRESPONSEOBJECT_H
#define ALITRDPIDRESPONSEOBJECT_H

#ifndef ROOT_TNamed
#include <TNamed.h>
#endif

#ifndef AliTRDPIDRESPONSE_H
#include "AliTRDPIDResponse.h"
#endif


class AliTRDPIDParams;
class AliTRDPIDReference;
class AliTRDPIDResponse;

class AliTRDPIDResponseObject : public TNamed{
public:
    enum ETRDPIDResponseObjectStatus {
	kIsOwner = BIT(14)
    };

    AliTRDPIDResponseObject();
    AliTRDPIDResponseObject(const char *name);
    AliTRDPIDResponseObject(const AliTRDPIDResponseObject &ref);
    AliTRDPIDResponseObject &operator=(const AliTRDPIDResponseObject &ref);
    virtual ~AliTRDPIDResponseObject();

    virtual void Print(Option_t *opt) const;

    void SetPIDParams(AliTRDPIDParams *params,AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D);
    void SetPIDReference(AliTRDPIDReference *params,AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D, Int_t NofCharges=1);

    // Derive reference
    TObject *GetLowerReference(AliPID::EParticleType spec, Float_t p, Float_t &pLower,AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D, Int_t Charge=0) const;
    TObject *GetUpperReference(AliPID::EParticleType spec, Float_t p, Float_t &pUpper,AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D, Int_t Charge=0) const;

    Int_t GetNumberOfMomentumBins(AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D) const;

    // Derive threshold params
    Bool_t GetThresholdParameters(Int_t ntracklets, Double_t efficiency, Double_t *params,Double_t centrality = -1,AliTRDPIDResponse::ETRDPIDMethod method=AliTRDPIDResponse::kLQ1D, Int_t charge=0) const;

    // Number of SlicesQ0
    Int_t GetNSlicesQ0() const{return fNSlicesQ0;}
    void SetNSlicesQ0(Int_t nsl){fNSlicesQ0=nsl;}

private:

      AliTRDPIDParams *fPIDParams[AliTRDPIDResponse::kNMethod]; // Contains Thresholds
      AliTRDPIDReference *fPIDReference[AliTRDPIDResponse::kNMethod]; // Contains References
      Int_t fNSlicesQ0; // Number of Slices for Q0

    ClassDef(AliTRDPIDResponseObject, 1);
};
#endif
