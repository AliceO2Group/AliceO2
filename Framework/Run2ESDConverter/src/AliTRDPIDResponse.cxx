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
// Calculation of probabilities using Bayesian approach
// Attention: This method is only used to separate electrons from pions
//
// Authors:
//  Markus Fasel <M.Fasel@gsi.de>
//  Anton Andronic <A.Andronic@gsi.de>
//
//  modifications 29.10. Yvonne Pachmayer <pachmay@physi.uni-heidelberg.de>
//
#include <TAxis.h>
#include <TClass.h>
#include <TDirectory.h>
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>
#include <TKey.h>
#include <TMath.h>
#include <TObjArray.h>
#include <TROOT.h> 
#include <TString.h>
#include <TSystem.h>

#include "AliLog.h"

#include "AliVTrack.h"

#include "AliTRDPIDResponseObject.h"
#include "AliTRDPIDResponse.h"
//#include "AliTRDTKDInterpolator.h"
#include "AliTRDNDFast.h"
#include "AliTRDdEdxParams.h"
#include "AliExternalTrackParam.h"



ClassImp(AliTRDPIDResponse)

//____________________________________________________________
AliTRDPIDResponse::AliTRDPIDResponse():
    TObject()
  ,fkPIDResponseObject(NULL)
  ,fkTRDdEdxParams(NULL)
  ,fGainNormalisationFactor(1.)
  ,fCorrectEta(kFALSE)
  ,fCorrectCluster(kFALSE)
  ,fCorrectCentrality(kFALSE)
  ,fCurrCentrality(-1.)
  ,fMagField(0.)
{
    //
    // Default constructor
    //
}

//____________________________________________________________
AliTRDPIDResponse::AliTRDPIDResponse(const AliTRDPIDResponse &ref):
    TObject(ref)
  ,fkPIDResponseObject(NULL)
  ,fkTRDdEdxParams(NULL)
  ,fGainNormalisationFactor(ref.fGainNormalisationFactor)
  ,fCorrectEta(kFALSE)
  ,fCorrectCluster(kFALSE)
  ,fCorrectCentrality(kFALSE)
  ,fCurrCentrality(-1.)
  ,fMagField(0.)
{
    //
    // Copy constructor
    //
}

//____________________________________________________________
AliTRDPIDResponse &AliTRDPIDResponse::operator=(const AliTRDPIDResponse &ref){
    //
    // Assignment operator
    //
    if(this == &ref) return *this;

    // Make copy
    TObject::operator=(ref);
    fGainNormalisationFactor = ref.fGainNormalisationFactor;
    fkPIDResponseObject = ref.fkPIDResponseObject;
    fkTRDdEdxParams = ref.fkTRDdEdxParams;
    fCorrectEta = ref.fCorrectEta;
    fCorrectCluster = ref.fCorrectCluster;
    fCorrectCentrality = ref.fCorrectCentrality;
    fCurrCentrality = ref.fCurrCentrality;
    fMagField   = ref.fMagField;

    return *this;
}

//____________________________________________________________
AliTRDPIDResponse::~AliTRDPIDResponse(){
    //
    // Destructor
    //
    if(IsOwner()) {
        delete fkPIDResponseObject;
        delete fkTRDdEdxParams;
        delete fhEtaCorr[0];
        for (Int_t i=0;i<3;i++) delete fhClusterCorr[i];
        delete fhCentralityCorr[0];
    }
}

//____________________________________________________________
Bool_t AliTRDPIDResponse::Load(const Char_t * filename){
    //
    // Load References into the toolkit
    //
    AliDebug(1, "Loading reference histos from root file");
    TDirectory *owd = gDirectory;// store old working directory

    if(!filename)
        filename = Form("%s/STEER/LQ1dRef_v1.root",gSystem->ExpandPathName("$ALICE_ROOT"));
    TFile *in = TFile::Open(filename);
    if(!in){
        AliError("Ref file not available.");
        return kFALSE;
    }

    gROOT->cd();
    fkPIDResponseObject = dynamic_cast<const AliTRDPIDResponseObject *>(in->Get("TRDPIDResponse")->Clone());
    in->Close(); delete in;
    owd->cd();
    SetBit(kIsOwner, kTRUE);
    AliDebug(2, Form("Successfully loaded References for %d Momentum bins", fkPIDResponseObject->GetNumberOfMomentumBins()));
    return kTRUE;
}

//_________________________________________________________________________
Bool_t AliTRDPIDResponse::SetEtaCorrMap(Int_t i,TH2D* hMap)
{
    //
    // Load map for TRD eta correction (a copy is stored and will be deleted automatically).
    // If hMap is 0x0,the eta correction will be disabled and kFALSE is returned.
    // If the map can be set, kTRUE is returned.
    //

    //    delete fhEtaCorr[0];

    if (!hMap) {
        fhEtaCorr[0] = 0x0;

        return kFALSE;
    }

    fhEtaCorr[0] = (TH2D*)(hMap->Clone());

    return kTRUE;
}

//_________________________________________________________________________
Bool_t AliTRDPIDResponse::SetClusterCorrMap(Int_t i,TH2D* hMap)
{
    //
    // Load map for TRD cluster correction (a copy is stored and will be deleted automatically).
    // If hMap is 0x0,the cluster correction will be disabled and kFALSE is returned.
    // If the map can be set, kTRUE is returned.
    //


    if (!hMap) {
        fhClusterCorr[i] = 0x0;

        return kFALSE;
    }

    fhClusterCorr[i] = (TH2D*)(hMap->Clone());

    return kTRUE;
}

//_________________________________________________________________________
Bool_t AliTRDPIDResponse::SetCentralityCorrMap(Int_t i,TH2D* hMap)
{
    //
    // Load map for TRD centrality correction (a copy is stored and will be deleted automatically).
    // If hMap is 0x0,the centrality correction will be disabled and kFALSE is returned.
    // If the map can be set, kTRUE is returned.
    //

    if (!hMap) {
        fhCentralityCorr[0] = 0x0;

        return kFALSE;
    }

    fhCentralityCorr[0] = (TH2D*)(hMap->Clone());

    return kTRUE;
}

//____________________________________________________________
Double_t AliTRDPIDResponse::GetEtaCorrection(const AliVTrack *track, Double_t bg) const
{
    //
    // eta correction
    //
    

    if (!fhEtaCorr[0]) {
        AliError(Form("Eta correction requested, but map not initialised for iterator:%i (usually via AliPIDResponse). Returning eta correction factor 1!",1));
        return 1.;

    }

    Double_t fEtaCorFactor=1;

    Int_t nch = track->GetTRDNchamberdEdx();
    Int_t iter=0;
    
    if (nch < 4) {
        AliError(Form("Eta correction requested for track with  = %i, no map available. Returning default eta correction factor = 1!", nch));
        return 1.;
    }
    
    Double_t tpctgl= 1.1;
    tpctgl=track->GetTPCTgl();



    if ((fhEtaCorr[iter]->GetBinContent(fhEtaCorr[iter]->FindBin(tpctgl,bg)) != 0)) {
        fEtaCorFactor= fhEtaCorr[iter]->GetBinContent(fhEtaCorr[iter]->FindBin(tpctgl,bg));
        return fEtaCorFactor;
    }  else
    {
        return 1;
    }


}


//____________________________________________________________
Double_t AliTRDPIDResponse::GetClusterCorrection(const AliVTrack *track, Double_t bg) const
{
    //
    // eta correction
    //

    Int_t nch = track->GetTRDNchamberdEdx();
    
    Double_t fClusterCorFactor=1;
    
    if (nch < 4) {
        AliError(Form("Cluster correction requested for track with  = %i, no map available. Returning default cluster correction factor = 1!", nch));
        return 1.;
    }

    Int_t offset =4;
    const Int_t iter = nch-offset;
    const Int_t ncls = track->GetTRDNclusterdEdx();

    if (!fhClusterCorr[iter]) {
        AliError(Form("Cluster correction requested, but map not initialised for iterator:%i (usually via AliPIDResponse). Returning cluster correction factor 1!",1));
        return 1.;
    }

    if ((fhClusterCorr[iter]->GetBinContent(fhClusterCorr[iter]->FindBin(ncls,bg)) != 0)) {
        fClusterCorFactor= fhClusterCorr[iter]->GetBinContent(fhClusterCorr[iter]->FindBin(ncls,bg));
        return fClusterCorFactor;
    }  else
    {
        return 1;
    }

}

//____________________________________________________________
Double_t AliTRDPIDResponse::GetCentralityCorrection(const AliVTrack *track, Double_t bg) const
{
    //
    // centrality correction
    //
    

    if (!fhCentralityCorr[0]) {
        //	AliInfo(Form("centrality correction requested, but map not initialised for iterator:%i (usually via AliPIDResponse). Returning centrality correction factor 1!",1));
        return 1.;

    }

    Double_t fCentralityCorFactor=1;

    Int_t nch = track->GetTRDNchamberdEdx();
    Int_t iter=0;
    
    if (nch < 4) {
        //        Ali(Form("Centrality correction requested for track with  = %i, no map available. Returning default centrality correction factor = 1!", nch));
        return 1.;
    }
    

    if(fCurrCentrality<0) return 1.;


    if ((fhCentralityCorr[iter]->GetBinContent(fhCentralityCorr[iter]->FindBin(fCurrCentrality,bg)) != 0)) {
        fCentralityCorFactor= fhCentralityCorr[iter]->GetBinContent(fhCentralityCorr[iter]->FindBin(fCurrCentrality,bg));
        return fCentralityCorFactor;
    }  else
    {
        return 1;
    }


}


//____________________________________________________________
Double_t AliTRDPIDResponse::GetNumberOfSigmas(const AliVTrack *track, AliPID::EParticleType type, Bool_t fCorrectEta, Bool_t fCorrectCluster, Bool_t fCorrectCentrality) const
{
    //
    //calculate the TRD nSigma
    //

    const Double_t badval = -9999;
    Double_t info[5]; for(Int_t i=0; i<5; i++){info[i]=badval;}
    const Double_t delta = GetSignalDelta(track, type, kFALSE, fCorrectEta, fCorrectCluster, fCorrectCentrality, info);

    const Double_t mean = info[0];
    const Double_t res = info[1];
    if(res<0){
        return badval;
    }

    const Double_t sigma = mean*res;
    const Double_t eps = 1e-12;
    return delta/(sigma + eps);
}

//____________________________________________________________
Double_t AliTRDPIDResponse::GetSignalDelta( const AliVTrack* track, AliPID::EParticleType type, Bool_t ratio/*=kFALSE*/, Bool_t fCorrectEta, Bool_t fCorrectCluster, Bool_t fCorrectCentrality, Double_t *info/*=0x0*/) const
{
    //
    //calculate the TRD signal difference w.r.t. the expected
    //output other information in info[]
    //

    const Double_t badval = -9999;

    if(!track){
        return badval;
    }

    Double_t pTRD = 0;
    Int_t pTRDNorm =0 ;
    for(Int_t ich=0; ich<6; ich++){
        if(track->GetTRDmomentum(ich)>0)
        {
            pTRD += track->GetTRDmomentum(ich);
            pTRDNorm++;
        }
    }

    if(pTRDNorm>0)
    {
        pTRD/=pTRDNorm;
    }
    else return badval;

    if(!fkTRDdEdxParams){
        AliDebug(3,"fkTRDdEdxParams null");
        return -99999;
    }

    const Double_t nch = track->GetTRDNchamberdEdx();
    const Double_t ncls = track->GetTRDNclusterdEdx();

    //  fkTRDdEdxParams->Print();

    const TVectorF meanvec = fkTRDdEdxParams->GetMeanParameter(type, nch, ncls,fCorrectEta);
    if(meanvec.GetNrows()==0){
        return badval;
    }

    const TVectorF resvec  = fkTRDdEdxParams->GetSigmaParameter(type, nch, ncls,fCorrectEta);
    if(resvec.GetNrows()==0){
        return badval;
    }

    const Float_t *meanpar = meanvec.GetMatrixArray();
    const Float_t *respar  = resvec.GetMatrixArray();

    //============================================================================================<<<<<<<<<<<<<



    const Double_t bg = pTRD/AliPID::ParticleMass(type);
    const Double_t expsig = MeandEdxTR(&bg, meanpar);

    if(info){
        info[0]= expsig;
        info[1]= ResolutiondEdxTR(&ncls, respar);
    }



    const Double_t eps = 1e-10;

    // eta asymmetry correction
    Double_t corrFactorEta = 1.0;

    if (fCorrectEta) {
        corrFactorEta = GetEtaCorrection(track,bg);
    }

    // cluster correction
    Double_t corrFactorCluster = 1.0;
    if (fCorrectCluster) {
        corrFactorCluster = GetClusterCorrection(track,bg);
    }


    // centrality correction
    Double_t corrFactorCentrality = 1.0;
    if (fCorrectCentrality) {
        corrFactorCentrality = GetCentralityCorrection(track,bg);
    }


    AliDebug(3,Form("TRD trunc PID expected signal %f exp. resolution %f bg %f nch %f ncls %f etcoron/off %i clustercoron/off %i centralitycoron/off %i nsigma %f ratio %f \n",
                    expsig,ResolutiondEdxTR(&ncls, respar),bg,nch,ncls,fCorrectEta,fCorrectCluster,fCorrectCentrality,(corrFactorEta*corrFactorCluster*corrFactorCentrality*track->GetTRDsignal())/(expsig + eps),
                    (corrFactorEta*corrFactorCluster*corrFactorCentrality*track->GetTRDsignal()) - expsig));


    if(ratio){
        return (corrFactorEta*corrFactorCluster*corrFactorCentrality*track->GetTRDsignal())/(expsig + eps);
    }
    else{
        return (corrFactorEta*corrFactorCluster*corrFactorCentrality*track->GetTRDsignal()) - expsig;
    }



}


Double_t AliTRDPIDResponse::ResolutiondEdxTR(const Double_t * xx,  const Float_t * par)
{
    //
    //resolution
    //npar=3
    //

    const Double_t ncls = xx[0];
    //  return par[0]+par[1]*TMath::Power(ncls, par[2]);
    return TMath::Sqrt(par[0]*par[0]+par[1]*par[1]/ncls);
}

Double_t AliTRDPIDResponse::MeandEdxTR(const Double_t * xx,  const Float_t * pin)
{
    //
    //ALEPH+LOGISTIC parametrization for dEdx+TR, in unit of MIP
    //npar = 8 = 3+5
    //

    Float_t ptr[4]={0};
    for(Int_t ii=0; ii<3; ii++){
        ptr[ii+1]=pin[ii];
    }
    return MeanTR(xx,ptr) + MeandEdx(xx,&(pin[3]));
}

Double_t AliTRDPIDResponse::MeanTR(const Double_t * xx,  const Float_t * par)
{
    //
    //LOGISTIC parametrization for TR, in unit of MIP
    //npar = 4
    //

    const Double_t bg = xx[0];
    const Double_t gamma = sqrt(1+bg*bg);

    const Double_t p0 = TMath::Abs(par[1]);
    const Double_t p1 = TMath::Abs(par[2]);
    const Double_t p2 = TMath::Abs(par[3]);

    const Double_t zz = TMath::Log(gamma);
    const Double_t tryield = p0/( 1 + TMath::Exp(-p1*(zz-p2)) );

    return par[0]+tryield;
}

Double_t AliTRDPIDResponse::MeandEdx(const Double_t * xx,  const Float_t * par)
{
    //
    //ALEPH parametrization for dEdx
    //npar = 5
    //

    const Double_t bg = xx[0];
    const Double_t beta = bg/TMath::Sqrt(1.+ bg*bg);

    const Double_t p0 = TMath::Abs(par[0]);
    const Double_t p1 = TMath::Abs(par[1]);

    const Double_t p2 = TMath::Abs(par[2]);

    const Double_t p3 = TMath::Abs(par[3]);
    const Double_t p4 = TMath::Abs(par[4]);

    const Double_t aa = TMath::Power(beta, p3);

    const Double_t bb = TMath::Log( p2 + TMath::Power(1./bg, p4) );

    return (p1-aa-bb)*p0/aa;

}


//____________________________________________________________
Int_t AliTRDPIDResponse::GetResponse(Int_t n, const Double_t * const dedx, const Float_t * const p, Double_t prob[AliPID::kSPECIES],ETRDPIDMethod PIDmethod,Bool_t kNorm, const AliVTrack *track) const
{
    //
    // Calculate TRD likelihood values for the track based on dedx and
    // momentum values. The likelihoods are calculated by query the
    // reference data depending on the PID method selected
    //
    // Input parameter :
    //   n - number of dedx slices/chamber
    //   dedx - array of dedx slices organized layer wise
    //   p - array of momentum measurements organized layer wise
    //
    // Return parameters
    //   prob - probabilities allocated by TRD for particle specis
    //   kNorm - switch to normalize probabilities to 1. By default true. If false return not normalized prob.
    //
    // Return value
    //   number of tracklets used for PID, 0 if no PID
    //
    AliDebug(3,Form(" Response for PID method: %d",PIDmethod));
    Int_t iCharge=AliPID::kNoCharge;
    if (track!=NULL){
        if (track->Charge()>0){
            iCharge=AliPID::kPosCharge;
        }
        else {
            iCharge=AliPID::kNegCharge;
        }
    }
    if(!fkPIDResponseObject){
        AliDebug(3,"Missing reference data. PID calculation not possible.");
        return 0;
    }

    for(Int_t is(AliPID::kSPECIES); is--;) prob[is]=.2;
    Double_t prLayer[AliPID::kSPECIES];
    Double_t dE[10], s(0.);
    Int_t ntrackletsPID=0;
    for(Int_t il(kNlayer); il--;){
        memset(prLayer, 0, AliPID::kSPECIES*sizeof(Double_t));
        if(!CookdEdx(n, &dedx[il*n], &dE[0],PIDmethod)) continue;
        s=0.;
        Bool_t filled=kTRUE;
        for(Int_t is(AliPID::kSPECIES); is--;){
            //if((PIDmethod==kLQ2D)&&(!(is==0||is==2)))continue;
            if((dE[0] > 0.) && (p[il] > 0.)) prLayer[is] = GetProbabilitySingleLayer(is, p[il], &dE[0],PIDmethod,iCharge);
            AliDebug(3, Form("Probability for Species %d in Layer %d: %e", is, il, prLayer[is]));
            if(prLayer[is]<1.e-30){
                AliDebug(2, Form("Null for species %d species prob layer[%d].",is,il));
                filled=kFALSE;
                break;
            }
            s+=prLayer[is];
        }
        if(!filled){
            continue;
        }
        if(s<1.e-30){
            AliDebug(2, Form("Null all species prob layer[%d].", il));
            continue;
        }
        for(Int_t is(AliPID::kSPECIES); is--;){
            if(kNorm) prLayer[is] /= s;  // probability in each layer for each particle species normalized to the sum of probabilities for given layer
            prob[is] *= prLayer[is];  // multiply single layer probabilities to get probability for complete track
        }
        ntrackletsPID++;
    }
    if(!kNorm) return ntrackletsPID;

    s=0.;
    // sum probabilities for all considered particle species
    for(Int_t is(AliPID::kSPECIES); is--;) { s+=prob[is];
    }
    if(s<1.e-30){
        AliDebug(2, "Null total prob.");
        return 0;
    }
    // norm to the summed probability  (default values: s=1 prob[is]=0.2)
    for(Int_t is(AliPID::kSPECIES); is--;){ prob[is]/=s; }
    return ntrackletsPID;
}

//____________________________________________________________
Double_t AliTRDPIDResponse::GetProbabilitySingleLayer(Int_t species, Double_t plocal, Double_t *dEdx,ETRDPIDMethod PIDmethod, Int_t Charge) const {
    //
    // Get the non-normalized probability for a certain particle species as coming
    // from the reference histogram
    // Interpolation between momentum bins
    //
    AliDebug(1, Form("Make Probability for Species %d with Momentum %f", species, plocal));

    Double_t probLayer = 0.;

    Float_t pLower, pUpper;

    AliTRDNDFast *refUpper = dynamic_cast<AliTRDNDFast *>(fkPIDResponseObject->GetUpperReference((AliPID::EParticleType)species, plocal, pUpper,PIDmethod,Charge)),
            *refLower = dynamic_cast<AliTRDNDFast *>(fkPIDResponseObject->GetLowerReference((AliPID::EParticleType)species, plocal, pLower,PIDmethod, Charge));
    if ((!refLower)&&(!refUpper)&&(Charge!=AliPID::kNoCharge)){
        AliDebug(3,Form("No references available for Charge; References for both Charges used"));
        refUpper = dynamic_cast<AliTRDNDFast *>(fkPIDResponseObject->GetUpperReference((AliPID::EParticleType)species, plocal, pUpper,PIDmethod,AliPID::kNoCharge));
        refLower = dynamic_cast<AliTRDNDFast *>(fkPIDResponseObject->GetLowerReference((AliPID::EParticleType)species, plocal, pLower,PIDmethod,AliPID::kNoCharge));
    }
    // Do Interpolation exept for underflow and overflow
    if(refLower && refUpper){
        Double_t probLower = refLower->Eval(dEdx);
        Double_t probUpper = refUpper->Eval(dEdx);

        probLayer = probLower + (probUpper - probLower)/(pUpper-pLower) * (plocal - pLower);
    } else if(refLower){
        // underflow
        probLayer = refLower->Eval(dEdx);
    } else if(refUpper){
        // overflow
        probLayer = refUpper->Eval(dEdx);
    } else {
        AliDebug(3,"No references available");
    }

    switch(PIDmethod){
    case kLQ2D: // 2D LQ
    {
        AliDebug(1,Form("Eval 2D Q0 %f Q1 %f P %e ",dEdx[0],dEdx[1],probLayer));
    }
        break;
    case kLQ1D: // 1D LQ
    {
        AliDebug(1, Form("Eval 1D dEdx %f Probability %e", dEdx[0],probLayer));
    }
        break;
    case kLQ3D: // 3D LQ
    {
        AliDebug(1, Form("Eval 1D dEdx %f %f %f Probability %e", dEdx[0],dEdx[1],dEdx[2],probLayer));
    }
        break;
    case kLQ7D: // 7D LQ
    {
        AliDebug(1, Form("Eval 1D dEdx %f %f %f %f %f %f %f Probability %e", dEdx[0],dEdx[1],dEdx[2],dEdx[3],dEdx[4],dEdx[5],dEdx[6],probLayer));
    }
        break;
    default:
        break;
    }

    return probLayer;

    /* old implementation

switch(PIDmethod){
case kNN: // NN
      break;
  case kLQ2D: // 2D LQ
      {
      if(species==0||species==2){ // references only for electrons and pions
          Double_t error = 0.;
          Double_t point[kNslicesLQ2D];
          for(Int_t idim=0;idim<kNslicesLQ2D;idim++){point[idim]=dEdx[idim];}

          AliTRDTKDInterpolator *refUpper = dynamic_cast<AliTRDTKDInterpolator *>(fkPIDResponseObject->GetUpperReference((AliPID::EParticleType)species, plocal, pUpper,kLQ2D)),
          *refLower = dynamic_cast<AliTRDTKDInterpolator *>(fkPIDResponseObject->GetLowerReference((AliPID::EParticleType)species, plocal, pLower,kLQ2D));
          // Do Interpolation exept for underflow and overflow
          if(refLower && refUpper){
                  Double_t probLower=0,probUpper=0;
          refLower->Eval(point,probLower,error);
                  refUpper->Eval(point,probUpper,error);
          probLayer = probLower + (probUpper - probLower)/(pUpper-pLower) * (plocal - pLower);
          } else if(refLower){
          // underflow
          refLower->Eval(point,probLayer,error);
          } else if(refUpper){
          // overflow
          refUpper->Eval(point,probLayer,error);
          } else {
          AliError("No references available");
          }
          AliDebug(2,Form("Eval 2D Q0 %f Q1 %f P %e Err %e",point[0],point[1],probLayer,error));
      }
      }
      break;
  case kLQ1D: // 1D LQ
      {
      TH1 *refUpper = dynamic_cast<TH1 *>(fkPIDResponseObject->GetUpperReference((AliPID::EParticleType)species, plocal, pUpper,kLQ1D)),
          *refLower = dynamic_cast<TH1 *>(fkPIDResponseObject->GetLowerReference((AliPID::EParticleType)species, plocal, pLower,kLQ1D));
      // Do Interpolation exept for underflow and overflow
      if(refLower && refUpper){
          Double_t probLower = refLower->GetBinContent(refLower->GetXaxis()->FindBin(dEdx[0]));
          Double_t probUpper = refUpper->GetBinContent(refUpper->GetXaxis()->FindBin(dEdx[0]));

          probLayer = probLower + (probUpper - probLower)/(pUpper-pLower) * (plocal - pLower);
      } else if(refLower){
          // underflow
          probLayer = refLower->GetBinContent(refLower->GetXaxis()->FindBin(dEdx[0]));
      } else if(refUpper){
          // overflow
          probLayer = refUpper->GetBinContent(refUpper->GetXaxis()->FindBin(dEdx[0]));
      } else {
          AliError("No references available");
      }
      AliDebug(1, Form("Eval 1D dEdx %f Probability %e", dEdx[0],probLayer));
      }
      break;
  default:
      break;
      }
      return probLayer;
      */

}

//____________________________________________________________
void AliTRDPIDResponse::SetOwner(){
    //
    // Make Deep Copy of the Reference Histograms
    //
    if(!fkPIDResponseObject || IsOwner()) return;
    const AliTRDPIDResponseObject *tmp = fkPIDResponseObject;
    fkPIDResponseObject = dynamic_cast<const AliTRDPIDResponseObject *>(tmp->Clone());
    SetBit(kIsOwner, kTRUE);
}

//____________________________________________________________
Bool_t AliTRDPIDResponse::CookdEdx(Int_t nSlice, const Double_t * const in, Double_t *out,ETRDPIDMethod PIDmethod) const
{
    //
    // Recalculate dE/dx
    // removed missing slices cut, detailed checks see presentation in TRD meeting: https://indico.cern.ch/event/506345/contribution/3/attachments/1239069/1821088/TRDPID_missingslices_charge.pdf
    //

    switch(PIDmethod){
    case kNN: // NN
        break;
    case kLQ2D: // 2D LQ
        out[0]=0;
        out[1]=0;
        for(Int_t islice = 0; islice < nSlice; islice++){
            //   if(in[islice]<=0){out[0]=0;out[1]=0;return kFALSE;}  // Require that all slices are filled

            if(islice<kNsliceQ0LQ2D)out[0]+= in[islice];
            else out[1]+= in[islice];
        }
        // normalize signal to number of slices
        out[0]*=1./Double_t(kNsliceQ0LQ2D);
        out[1]*=1./Double_t(nSlice-kNsliceQ0LQ2D);
        if(out[0] <= 0) return kFALSE;
        if(out[1] <= 0) return kFALSE;
        AliDebug(3,Form("CookdEdx Q0 %f Q1 %f",out[0],out[1]));
        break;
    case kLQ1D: // 1D LQ
        out[0]= 0.;
        for(Int_t islice = 0; islice < nSlice; islice++) {
            //	  if(in[islice] > 0) out[0] += in[islice] * fGainNormalisationFactor;   // Protect against negative values for slices having no dE/dx information
            if(in[islice] > 0) out[0] += in[islice];  // no neg dE/dx values
        }
        out[0]*=1./Double_t(kNsliceQ0LQ1D);
        if(out[0] <= 0) return kFALSE;
        AliDebug(3,Form("CookdEdx dEdx %f",out[0]));
        break;
    case kLQ3D: // 3D LQ
        out[0]=0;
        out[1]=0;
        out[2]=0;
        for(Int_t islice = 0; islice < nSlice; islice++){
            // if(in[islice]<=0){out[0]=0;out[1]=0;out[2]=0;return kFALSE;}  // Require that all slices are filled
            if(islice<kNsliceQ0LQ3D)out[0]+= in[islice];
            out[1]=(in[3]+in[4]);
            out[2]=(in[5]+in[6]);
        }
        // normalize signal to number of slices
        out[0]*=1./Double_t(kNsliceQ0LQ3D);
        out[1]*=1./2.;
        out[2]*=1./2.;
        if(out[0] <= 0) return kFALSE;
        if(out[1] <= 0) return kFALSE;
        if(out[2] <= 0) return kFALSE;
        AliDebug(3,Form("CookdEdx Q0 %f Q1 %f Q2 %f",out[0],out[1],out[2]));
        break;
    case kLQ7D: // 7D LQ
        for(Int_t i=0;i<nSlice;i++) {out[i]=0;}
        for(Int_t islice = 0; islice < nSlice; islice++){
            if(in[islice]<=0){
                for(Int_t i=0;i<8;i++){
                    out[i]=0;
                }
                return kFALSE;}  // Require that all slices are filled
            out[islice]=in[islice];
        }
        for(Int_t i=0;i<nSlice;i++) {if(out[i]<=0) return kFALSE; }
        AliDebug(3,Form("CookdEdx Q0 %f Q1 %f Q2 %f Q3 %f Q4 %f Q5 %f Q6 %f Q7 %f",out[0],out[1],out[2],out[3],out[4],out[5],out[6],out[7]));
        break;

    default:
        return kFALSE;
    }
    return kTRUE;
}

//____________________________________________________________
Bool_t AliTRDPIDResponse::IdentifiedAsElectron(Int_t nTracklets, const Double_t *like, Double_t p, Double_t level,Double_t centrality,ETRDPIDMethod PIDmethod, const AliVTrack *vtrack) const {
    //
    // Check whether particle is identified as electron assuming a certain electron efficiency level
    // Only electron and pion hypothesis is taken into account
    //
    // Inputs:
    //         Number of tracklets
    //         Likelihood values
    //         Momentum
    //         Electron efficiency level
    //
    // If the function fails when the params are not accessible, the function returns true
    //
    Int_t iCharge=AliPID::kNoCharge;
    if (vtrack!=NULL){
        Int_t vTrCharge=vtrack->Charge();
        if (vTrCharge>0){
            iCharge=AliPID::kPosCharge;
        }
        else{
            iCharge=AliPID::kNegCharge;
        }
    }
    if(!fkPIDResponseObject){
        AliDebug(3,"No PID Param object available");
        return kTRUE;
    }
    Double_t probEle = like[AliPID::kElectron]/(like[AliPID::kElectron] + like[AliPID::kPion]);
    AliDebug(3,Form("probabilities like %f %f %f \n",probEle,like[AliPID::kElectron],like[AliPID::kPion]));
    Double_t params[4];
    if(!fkPIDResponseObject->GetThresholdParameters(nTracklets, level, params,centrality,PIDmethod,iCharge)){
        //AliError("No Params found for the given configuration with chosen Charge");
        //AliError("Using Parameters for both charges");
        if((iCharge!=AliPID::kNoCharge)&&(!fkPIDResponseObject->GetThresholdParameters(nTracklets, level, params,centrality,PIDmethod,AliPID::kNoCharge))){
            //AliError("No Params found for the given configuration with charge 0");
            return kTRUE;
        }
        if(iCharge==AliPID::kNoCharge){
            return kTRUE;
        }
    }

    Double_t threshold = 1. - params[0] - params[1] * p - params[2] * TMath::Exp(-params[3] * p);
    AliDebug(3,Form("is ident details %i %f %f %i %f %f %f %f \n",nTracklets, level, centrality,PIDmethod,probEle, threshold,TMath::Min(threshold, 0.99),TMath::Max(TMath::Min(threshold, 0.99), 0.2)));
    if(probEle > TMath::Max(TMath::Min(threshold, 0.99), 0.2)) return kTRUE;  // truncate the threshold upperwards to 0.999 and lowerwards to 0.2 and exclude unphysical values
    return kFALSE;
}

//____________________________________________________________
Bool_t AliTRDPIDResponse::SetPIDResponseObject(const AliTRDPIDResponseObject * obj){

    fkPIDResponseObject = obj;
    if((AliLog::GetDebugLevel("",IsA()->GetName()))>0 && obj) fkPIDResponseObject->Print("");
    return kTRUE;
}


//____________________________________________________________
Bool_t AliTRDPIDResponse::SetdEdxParams(const AliTRDdEdxParams * par)
{
    fkTRDdEdxParams = par;
    return kTRUE;
}
