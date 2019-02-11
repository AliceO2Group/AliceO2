#ifndef AliEMCALPIDResponse_h
#define AliEMCALPIDResponse_h

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliEMCALPIDResponse                                                  //
//                                                                      //
// EMCAL class to perfom PID                                            //
// This is a prototype and still under development                      //
//                                                                      //
// Author: Michael Weber (m.weber@cern.ch)                              //
//////////////////////////////////////////////////////////////////////////

#include "AliPID.h"
#include <TVectorD.h>

class TF1;

class AliEMCALPIDResponse: public TObject 
{
public : 
    AliEMCALPIDResponse();    //ctor
    AliEMCALPIDResponse( const AliEMCALPIDResponse& other);                //copy ructor
    AliEMCALPIDResponse &operator=( const AliEMCALPIDResponse& other);     //assignment operator

    virtual ~AliEMCALPIDResponse();     //dtor
  

    // Getters
    Double_t  GetNumberOfSigmas( Float_t pt,  Float_t eop, AliPID::EParticleType n,  Int_t charge) const;
    Double_t  GetExpectedNorm  ( Float_t pt, AliPID::EParticleType n,  Int_t charge) const;
  
    //Setters
    void   SetPIDParams(const TObjArray * params) { fkPIDParams = params; }
    void   SetCentrality(Float_t currentCentrality) { fCurrCentrality = currentCentrality;}
    

    // EMCAL probability
    Bool_t ComputeEMCALProbability(Int_t nSpecies, Float_t pt, Float_t eop, Int_t charge, Double_t *pEMCAL) const;

protected:
  
private:

  TF1 *fNorm;                            // Gauss function for normalizing NON electron probabilities 

  Double_t fCurrCentrality;              // current (in the current event) centrality percentile 

  const TObjArray *fkPIDParams;               // PID Params

  const TVectorD* GetParams(Int_t nParticle, Float_t fPt, Int_t charge) const; 

  ClassDef(AliEMCALPIDResponse, 2)
};

#endif // #ifdef AliEMCALPIDResponse_cxx

