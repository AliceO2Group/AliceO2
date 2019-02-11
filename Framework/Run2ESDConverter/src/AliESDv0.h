#ifndef ALIESDV0_H
#define ALIESDV0_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//                          ESD V0 Vertex Class
//          This class is part of the Event Summary Data set of classes
//    Origin: Iouri Belikov, CERN, Jouri.Belikov@cern.ch
//    Modified by: Marian Ivanov,  CERN, Marian.Ivanov@cern.ch
//            and  Boris Hippolyte,IPHC, hippolyt@in2p3.fr 
//-------------------------------------------------------------------------

#include <TPDGCode.h>

#include "AliExternalTrackParam.h"
#include "AliVParticle.h"

class AliESDV0Params;

class AliESDv0 : public AliVParticle {
public:
  enum {kUsedByCascadeBit = BIT(14)};
  AliESDv0();
  AliESDv0(const AliExternalTrackParam &t1, Int_t i1,
           const AliExternalTrackParam &t2, Int_t i2);

  AliESDv0(const AliESDv0& v0);
  virtual ~AliESDv0();
  Int_t Refit();
  AliESDv0& operator=(const AliESDv0& v0);
  virtual void Copy(TObject &obj) const;

// Start with AliVParticle functions
  virtual Double_t Px() const { return fNmom[0]+fPmom[0]; }
  virtual Double_t Py() const { return fNmom[1]+fPmom[1]; }
  virtual Double_t Pz() const { return fNmom[2]+fPmom[2]; }
  virtual Double_t Pt() const { return TMath::Sqrt(Px()*Px()+Py()*Py()); }
  virtual Double_t P()  const { 
     return TMath::Sqrt(Px()*Px()+Py()*Py()+Pz()*Pz()); 
  }
  virtual Bool_t   PxPyPz(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
  virtual Double_t Xv() const { return fPos[0]; }
  virtual Double_t Yv() const { return fPos[1]; }
  virtual Double_t Zv() const { return fPos[2]; }
  virtual Bool_t   XvYvZv(Double_t x[3]) const { x[0] = Xv(); x[1] = Yv(); x[2] = Zv(); return kTRUE; }
  virtual Double_t OneOverPt() const { return (Pt() != 0.) ? 1./Pt() : -999.; }
  virtual Double_t Phi() const {return TMath::Pi()+TMath::ATan2(-Py(),-Px()); }
  virtual Double_t Theta() const {return 0.5*TMath::Pi()-TMath::ATan(Pz()/(Pt()+1.e-13)); }
  virtual Double_t E() const; // default is KOs but can be changed via ChangeMassHypothesis (defined in the .cxx)
  virtual Double_t M() const { return GetEffMass(); }
  virtual Double_t Eta() const { return 0.5*TMath::Log((P()+Pz())/(P()-Pz()+1.e-13)); }
  virtual Double_t Y() const;
  virtual Short_t  Charge() const { return 0; }
  virtual Int_t    GetLabel() const { return -1; }  // temporary
  virtual const Double_t *PID() const { return 0; } // return PID object ? (to be discussed!)
  
  // Then extend the AliVParticle functions
  Double_t E(Int_t pdg) const;
  Double_t Y(Int_t pdg) const;

  // Now the functions for analysis consistency
  Double_t RapK0Short() const;
  Double_t RapLambda() const;
  Double_t AlphaV0() const;
  Double_t PtArmV0() const;

  // Eventually the older functions
  Double_t ChangeMassHypothesis(Int_t code=kK0Short); 

  Int_t    GetPdgCode() const {return fPdgCode;}
  Double_t  GetEffMass(UInt_t p1, UInt_t p2) const;
  Double_t  GetEffMassExplicit(Double_t m1, Double_t m2) const;
  Double_t  GetEffMass() const {return fEffMass;}
  Double_t  GetChi2V0()  const {return fChi2V0;}
  void     GetPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const;
  void     GetNPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const;
  void     GetPPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const;  
  void     GetXYZ(Double_t &x, Double_t &y, Double_t &z) const;
  Float_t  GetD(Double_t x0,Double_t y0) const;
  Float_t  GetD(Double_t x0,Double_t y0,Double_t z0) const;
  Int_t    GetNindex() const {return fNidx;}
  Int_t    GetPindex() const {return fPidx;}
  void     SetDcaV0Daughters(Double_t rDcaV0Daughters=0.);
  Double_t GetDcaV0Daughters() const {return fDcaV0Daughters;}
  Float_t  GetV0CosineOfPointingAngle(Double_t refPointX, Double_t refPointY, Double_t refPointZ) const;
  Double_t GetV0CosineOfPointingAngle() const {return fPointAngle;}
  void     SetV0CosineOfPointingAngle(Double_t cpa) {fPointAngle=cpa;}
  void     SetOnFlyStatus(Bool_t status){fOnFlyStatus=status;}
  Bool_t   GetOnFlyStatus() const {return fOnFlyStatus;}
  const AliExternalTrackParam *GetParamP() const {return &fParamP;}
  const AliExternalTrackParam *GetParamN() const {return &fParamN;}
  AliESDVertex GetVertex() const;


  // **** The following member functions need to be revised ***

  void GetPosCov(Double_t cov[6])const ; // getter for the covariance matrix of the V0 position 
  Double_t GetSigmaY();     // sigma of y coordinate at vertex posistion
  Double_t GetSigmaZ();     // sigma of z coordinate at vertex posistion
  Double_t GetSigmaAP0();   // calculate sigma of Point angle resolution at vertex pos.
  Double_t GetSigmaD0();    // calculate sigma of position resolution at vertex pos.
  Double_t GetEffectiveSigmaAP0();   // calculate sigma of point angle resolution at vertex pos. effecive parameterization
  Double_t GetEffectiveSigmaD0();    // calculate sigma of position resolution at vertex pos.
  Double_t GetMinimaxSigmaAP0();    // calculate mini-max sigma of point angle resolution
  Double_t GetMinimaxSigmaD0();     // calculate mini-max sigma of dca resolution
  Double_t GetLikelihoodAP(Int_t mode0, Int_t mode1);   // get likelihood for point angle
  Double_t GetLikelihoodD(Int_t mode0, Int_t mode1);    // get likelihood for DCA
  Double_t GetLikelihoodC(Int_t mode0, Int_t mode1) const;    // get likelihood for Causality
  //
  //
  static const AliESDV0Params & GetParameterization(){return fgkParams;}
  void SetParamP(const AliExternalTrackParam & paramP) {fParamP = paramP;}
  void SetParamN(const AliExternalTrackParam & paramN) {fParamN = paramN;}
  void SetStatus(Int_t status){fStatus=status;}
  ULong64_t GetStatus() const {return ULong64_t(fStatus);}
  Int_t GetIndex(Int_t i) const {return (i==0) ? fNidx : fPidx;}
  void SetIndex(Int_t i, Int_t ind);
  const Double_t *GetAnglep() const {return fAngle;}
  Double_t GetRr() const {return fRr;}
  Double_t GetDistSigma() const {return fDistSigma;}
  void SetDistSigma(Double_t ds) {fDistSigma=ds;}
  Float_t GetChi2Before() const {return fChi2Before;}
  void SetChi2Before(Float_t cb) {fChi2Before=cb;}
  Float_t GetChi2After() const {return fChi2After;}
  void SetChi2After(Float_t ca) {fChi2After=ca;}
  Float_t GetNAfter() const {return fNAfter;}
  void SetNAfter(Short_t na) {fNAfter=na;}
  Short_t GetNBefore() const {return fNBefore;}
  void SetNBefore(Short_t nb) {fNBefore=nb;}  
  void SetCausality(Float_t pb0, Float_t pb1, Float_t pa0, Float_t pa1);
  const Double_t * GetCausalityP() const {return fCausality;}
  void SetClusters(const Int_t *clp, const Int_t *clm);
  const Int_t * GetClusters(Int_t i) const {return fClusters[i];}
  void SetNormDCAPrim(Float_t nd0, Float_t nd1){fNormDCAPrim[0] = nd0; fNormDCAPrim[1]=nd1;}
  const Double_t  *GetNormDCAPrimP() const {return fNormDCAPrim;}
    // Dummy
  Int_t    PdgCode() const {return 0;}
  
  //virtual Bool_t   GetPxPyPz(Double_t */*p*/) const { return kFALSE; }
  virtual void     SetID(Short_t /*id*/) {;}
  Double_t GetKFInfo(UInt_t p1, UInt_t p2, Int_t type) const;
  Double_t GetKFInfoScale(UInt_t p1, UInt_t p2, Int_t type, Double_t d1pt, Double_t s1pt, Double_t eLoss=0, Int_t flag=0x3) const;
  //
  void SetUsedByCascade(Bool_t v) {SetBit(kUsedByCascadeBit,v);}
  Bool_t GetUsedByCascade() const {return TestBit(kUsedByCascadeBit);}
  
protected:
  AliExternalTrackParam fParamN;  // external parameters of negative particle
  AliExternalTrackParam fParamP;  // external parameters of positive particle

  // CKBrev: tkink about revision

  Double32_t   fEffMass;          // reconstructed V0's effective mass
  Double32_t   fDcaV0Daughters;   // dca between V0's daughters
  Double32_t   fChi2V0;           // V0's chi2 value
  Double32_t   fPos[3];         // V0's position (global)
  Double32_t   fPosCov[6];      // covariance matrix of the vertex position
  Double32_t   fNmom[3];        // momentum of the negative daughter (global)
  Double32_t   fPmom[3];        // momentum of the positive daughter (global)
  Double32_t   fNormDCAPrim[2];  // normalize distance to the primary vertex CKBrev
  Double32_t   fRr;         //rec position of the vertex CKBrev
  Double32_t   fDistSigma; //sigma of distance CKBrev
  Double32_t        fChi2Before;   //chi2 of the tracks before V0 CKBrev
  Double32_t        fChi2After;   // chi2 of the tracks after V0 CKBrev


  Double32_t        fCausality[4]; //[0,1,8] causality information - see comments in SetCausality CKBrev
  Double32_t        fAngle[3];   //[-2*pi,2*pi,16]three angles CKBrev
  Double32_t        fPointAngleFi; //[-1,1,16]point angle fi CKBrev
  Double32_t        fPointAngleTh; //[-1,1,16]point angle theta CKBrev
  Double32_t        fPointAngle;   //[-1,1,32] cosine of the pointing angle


  Int_t fPdgCode;             // reconstructed V0's type (PDG code)
  Int_t fClusters[2][6];      //! its clusters CKBrev  
  Int_t fNidx;                // index of the negative daughter
  Int_t fPidx;                // index of the positive daughter



  Short_t    fStatus;     //status CKBrev
  Short_t    fNBefore;      // number of possible points before V0 CKBrev
  Short_t    fNAfter;      // number of possible points after V0 CKBrev

  Bool_t     fOnFlyStatus;    // if kTRUE, then this V0 is recontructed
                            // "on fly" during the tracking

  //
  // parameterization coefficients
  static const AliESDV0Params fgkParams;  //! resolution and likelihood parameterization  

private:

  ClassDef(AliESDv0,6)      // ESD V0 vertex
};

inline 
void AliESDv0::GetNPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const {
px=fNmom[0]; py=fNmom[1]; pz=fNmom[2];
}

inline 
void AliESDv0::GetPPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const {
px=fPmom[0]; py=fPmom[1]; pz=fPmom[2];
}

inline
void AliESDv0::SetDcaV0Daughters(Double_t rDcaV0Daughters){
  fDcaV0Daughters=rDcaV0Daughters;
}

inline 
void AliESDv0::SetIndex(Int_t i, Int_t ind) {
  if(i==0)
    fNidx=ind;
  else
    fPidx=ind;
}

#endif
