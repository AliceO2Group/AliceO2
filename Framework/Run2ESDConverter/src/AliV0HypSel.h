#ifndef ALIV0HYPSEL_H
#define ALIV0HYPSEL_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

//------------------------------------------------------------------
//                    V0 hypothesis selection params
// Used to check if given V0 is candidate for given hypothesis
//------------------------------------------------------------------

#include "TNamed.h"


//_____________________________________________________________________________
class AliV0HypSel : public TNamed {
public:

  AliV0HypSel();
  AliV0HypSel(const AliV0HypSel& src);
  AliV0HypSel(const char *name, float m0,float m1, float mass, float sigma, 
	      float nsig, float margin, float cf0=0., float cf1=0.);
  void Validate();

  float GetM0()     const {return fM0;}
  float GetM1()     const {return fM1;}
  float GetMass()   const {return fMass;}
  float GetSigmaM() const {return fSigmaM;}
  float GetNSigma() const {return fNSigma;}
  float GetCoef0Pt() const {return fCoef0Pt;}
  float GetCoef1Pt() const {return fCoef1Pt;}
  float GetMarginAdd() const {return fMarginAdd;}
  float GetMassMargin(float pT) const {return fNSigma*fgBFieldCoef*fSigmaM*(fCoef0Pt+pT*fCoef1Pt)+fMarginAdd;}

  static void  AccountBField(float b);
  static void  SetBFieldCoef(float v) {fgBFieldCoef = v>0. ? v : 1.0;}
  static float GetBFieldCoef() { return fgBFieldCoef; }
  
  virtual void Print(const Option_t *) const;
  
private:
  Float_t fM0;         // mass of the 1st prong
  Float_t fM1;         // mass of the 2nd prong
  Float_t fMass ;      // expected V0 mass
  Float_t fSigmaM;     // rough sigma estimate for sigmaMass = fSigmaM*(fCoef0Pt+fCoef1Pt*Pt) parameterization
  Float_t fCoef0Pt;    // offset of sigma_m pT dependence
  Float_t fCoef1Pt;    // pT proportional coef. of sigma_m pT dependence  
  Float_t fNSigma;     // number fSigmaM to apply
  Float_t fMarginAdd;  // additional additive safety margin

  static float fgBFieldCoef; // scaling of sigma due to non-nominal field
  ClassDef(AliV0HypSel,1)  // V0 Hypothesis selection
};


#endif


