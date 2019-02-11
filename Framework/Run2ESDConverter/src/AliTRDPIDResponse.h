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
// PID Response class for the TRD detector
// Based on 1D Likelihood approach
// For further information see implementation file
//
#ifndef ALITRDPIDRESPONSE_H
#define ALITRDPIDRESPONSE_H

#ifndef ROOT_TObject
#include <TObject.h>
#endif

#ifndef ALIPID_H
#include "AliPID.h"
#endif

class TObjArray;
class AliVTrack;
class AliTRDPIDResponseObject;
class AliTRDdEdxParams;
class TH2;
class TH2D;

class AliTRDPIDResponse : public TObject {
  public:
    enum ETRDPIDResponseStatus {
      kIsOwner = BIT(14)
    };
    enum ETRDPIDResponseDef {
	kNlayer = 6
       ,kNPBins = 6
    };
    enum ETRDPIDMethod {
	kNN   = 0,
	kLQ2D = 1,
	kLQ1D = 2,
	kLQ3D = 3,
	kLQ7D = 4
    };
    enum ETRDPIDNMethod {
	kNMethod=5
    };
    enum ETRDNslices {
	kNslicesLQ1D = 1,
	kNslicesLQ2D = 2,
	kNslicesNN = 7,
	kNslicesLQ3D = 3,
	kNslicesLQ7D = 7
    };
    enum ETRDNsliceQ0 {
	kNsliceQ0LQ1D = 1,
	kNsliceQ0LQ2D = 4,
	kNsliceQ0NN = 1,
	kNsliceQ0LQ3D = 3,
	kNsliceQ0LQ7D = 1
    };

    AliTRDPIDResponse();
    AliTRDPIDResponse(const AliTRDPIDResponse &ref);
    AliTRDPIDResponse& operator=(const AliTRDPIDResponse &ref);
    ~AliTRDPIDResponse();
    
    Double_t GetNumberOfSigmas(const AliVTrack *track, AliPID::EParticleType type, Bool_t fCorrectEta, Bool_t fCorrectCluster, Bool_t fCorrectCentrality) const;
    Double_t GetSignalDelta(const AliVTrack *track, AliPID::EParticleType type, Bool_t ratio=kFALSE, Bool_t fCorrectEta=kFALSE, Bool_t fCorrectCluster=kFALSE, Bool_t fCorrectCentrality=kFALSE, Double_t *info=0x0) const;

    static Double_t MeandEdx(const Double_t * xx, const Float_t * par);
    static Double_t MeanTR(const Double_t * xx, const Float_t * par);
    static Double_t MeandEdxTR(const Double_t * xx, const Float_t * par);
    static Double_t ResolutiondEdxTR(const Double_t * xx,  const Float_t * par);

    Int_t    GetResponse(Int_t n, const Double_t * const dedx, const Float_t * const p, Double_t prob[AliPID::kSPECIES],ETRDPIDMethod PIDmethod=kLQ1D, Bool_t kNorm=kTRUE, const AliVTrack *track=NULL) const;
    inline ETRDNslices  GetNumberOfSlices(ETRDPIDMethod PIDmethod=kLQ1D) const;

    Bool_t    IsOwner() const {return TestBit(kIsOwner);}
    
    void      SetOwner();
    void      SetGainNormalisationFactor(Double_t gainFactor) { fGainNormalisationFactor = gainFactor; }

    Bool_t SetPIDResponseObject(const AliTRDPIDResponseObject * obj);
    Bool_t SetdEdxParams(const AliTRDdEdxParams * par);
    
    // eta correction map
    TH2D* GetEtaCorrMap(Int_t n) const { return fhEtaCorr[n]; };
    Bool_t SetEtaCorrMap(Int_t n, TH2D* hMapn);

    // cluster correction map
    TH2D* GetClusterCorrMap(Int_t n) const { return fhClusterCorr[n]; };
    Bool_t SetClusterCorrMap(Int_t n, TH2D* hMapn);

    // centrality correction map
    TH2D* GetCentralityCorrMap(Int_t n) const { return fhCentralityCorr[n]; };
    Bool_t SetCentralityCorrMap(Int_t n, TH2D* hMapn);
    void  SetCentrality(Float_t currentCentrality) { fCurrCentrality = currentCentrality;}

    Bool_t    Load(const Char_t *filename = NULL);
  
    Bool_t    IdentifiedAsElectron(Int_t nTracklets, const Double_t *like, Double_t p, Double_t level,Double_t centrality=-1,ETRDPIDMethod PIDmethod=kLQ1D,const AliVTrack *vtrack=NULL) const;
    
    Double_t GetEtaCorrection(const AliVTrack *track, Double_t bg) const;
    Double_t GetClusterCorrection(const AliVTrack *track, Double_t bg) const;
    Double_t GetCentralityCorrection(const AliVTrack *track, Double_t bg) const;
    void     SetMagField(Double_t mf) { fMagField=mf; }
  
  private:
    Bool_t    CookdEdx(Int_t nSlice, const Double_t * const in, Double_t *out,ETRDPIDMethod PIDmethod=kLQ1D) const;
    Double_t  GetProbabilitySingleLayer(Int_t species, Double_t plocal, Double_t *dEdx,ETRDPIDMethod PIDmethod=kLQ1D, Int_t Charge=0) const;
    
    const AliTRDPIDResponseObject *fkPIDResponseObject;   //! PID References and Params
    const AliTRDdEdxParams * fkTRDdEdxParams; //! parametrisation for truncated mean
    Double_t  fGainNormalisationFactor;         //! Gain normalisation factor
    Bool_t    fCorrectEta;   //! switch for eta correction
    TH2D*     fhEtaCorr[1]; //! Map for TRD eta correction
    Bool_t    fCorrectCluster;   //! switch for cluster correction
    TH2D*     fhClusterCorr[3]; //! Map for TRD eta correction
    Bool_t    fCorrectCentrality;   //! switch for centrality correction
    TH2D*     fhCentralityCorr[1]; //! Map for TRD centrality correction
    Double_t fCurrCentrality;              // current (in the current event) centrality percentile

    Double_t  fMagField;  //! Magnetic field
  
  ClassDef(AliTRDPIDResponse, 8)    // Tool for TRD PID
};

AliTRDPIDResponse::ETRDNslices AliTRDPIDResponse::GetNumberOfSlices(ETRDPIDMethod PIDmethod) const {
  // Get the current number of slices
  ETRDNslices slices = kNslicesLQ1D;
  switch(PIDmethod){
    case kLQ1D: slices = kNslicesLQ1D; break;
    case kLQ2D: slices = kNslicesLQ2D; break;
    case kNN:   slices = kNslicesNN; break;
    case kLQ3D: slices = kNslicesLQ3D; break;
    case kLQ7D: slices = kNslicesLQ7D; break;
  };
  return slices;
}
#endif

