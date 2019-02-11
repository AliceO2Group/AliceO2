#ifndef ALIESDCASCADE_H
#define ALIESDCASCADE_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//-------------------------------------------------------------------------
//                        ESD Cascade Vertex Class
//               Implementation of the cascade vertex class
//    Origin: Christian Kuhn, IReS, Strasbourg, christian.kuhn@ires.in2p3.fr
//     Modified by: Antonin Maire,IPHC, Antonin.Maire@ires.in2p3.fr
//            and  Boris Hippolyte,IPHC, hippolyt@in2p3.fr 
//-------------------------------------------------------------------------

#include <TPDGCode.h>
#include "AliESDv0.h"

class AliLog;
class AliExternalTrackParam;

class AliESDcascade : public AliESDv0 {

public:
  AliESDcascade();
  AliESDcascade(const AliESDcascade& cas);
  AliESDcascade(const AliESDv0 &v0,
                const AliExternalTrackParam &t, Int_t i);
  ~AliESDcascade();
  AliESDcascade& operator=(const AliESDcascade& cas);
  virtual void Copy(TObject &obj) const;
    
  Int_t RefitCascade(AliExternalTrackParam *bachelor); //to be invoked immediately after creation

// Start with AliVParticle functions
  virtual Double_t Px() const { return fNmom[0]+fPmom[0]+fBachMom[0]; }
  virtual Double_t Py() const { return fNmom[1]+fPmom[1]+fBachMom[1]; }
  virtual Double_t Pz() const { return fNmom[2]+fPmom[2]+fBachMom[2]; }
  virtual Double_t Pt() const { return TMath::Sqrt(Px()*Px()+Py()*Py()); }
  virtual Double_t P()  const { 
     return TMath::Sqrt(Px()*Px()+Py()*Py()+Pz()*Pz()); 
  }
  virtual Bool_t   PxPyPz(Double_t p[3]) const { p[0] = Px(); p[1] = Py(); p[2] = Pz(); return kTRUE; }
  virtual Double_t Xv() const { return fPosXi[0]; }
  virtual Double_t Yv() const { return fPosXi[1]; }
  virtual Double_t Zv() const { return fPosXi[2]; }
  virtual Bool_t   XvYvZv(Double_t x[3]) const { x[0] = Xv(); x[1] = Yv(); x[2] = Zv(); return kTRUE; }
  virtual Double_t OneOverPt() const { return (Pt() != 0.) ? 1./Pt() : -999.; }
  virtual Double_t Phi() const {return TMath::Pi()+TMath::ATan2(-Py(),-Px()); }
  virtual Double_t Theta() const {return 0.5*TMath::Pi()-TMath::ATan(Pz()/(Pt()+1.e-13)); }
  virtual Double_t E() const; // default is Xis but can be changed via ChangeMassHypothesis (defined in the .cxx)
  virtual Double_t M() const { return GetEffMassXi(); }
  virtual Double_t Eta() const { return 0.5*TMath::Log((P()+Pz())/(P()-Pz()+1.e-13)); }
  virtual Double_t Y() const;
  virtual Short_t  Charge() const { return (GetPdgCodeXi()>0) ? -1 : 1; } // due to PDG sign convention !
  virtual Int_t    GetLabel() const { return -1; }  // temporary
  virtual const Double_t *PID() const { return 0; } // return PID object ? (to be discussed!)

  // Then extend the AliVParticle functions
  Double_t E(Int_t pdg) const;
  Double_t Y(Int_t pdg) const;

  // Now the functions for analysis consistency
  Double_t RapXi() const;
  Double_t RapOmega() const;
  Double_t AlphaXi() const;
  Double_t PtArmXi() const;

  // Eventually the older functions
  Double_t ChangeMassHypothesis(Double_t &v0q, Int_t code=kXiMinus); 

  Int_t    GetPdgCodeXi() const {return fPdgCodeXi;}
  Double_t GetEffMassXi() const {return fEffMassXi;}
  Double_t GetChi2Xi()  const {return fChi2Xi;}
  void     GetPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const;
  void     GetXYZcascade(Double_t &x, Double_t &y, Double_t &z) const;
  void     SetXYZcascade(Double_t x, Double_t y, Double_t z); //for testing purposes
  Double_t GetDcascade(Double_t x0, Double_t y0, Double_t z0) const;

  void     GetBPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const {
     px=fBachMom[0]; py=fBachMom[1]; pz=fBachMom[2];
  }

  Int_t    GetBindex() const {return fBachIdx;}
  void     SetIndex(Int_t i) {fBachIdx=i;}        //for the consistency with V0
  Int_t    GetIndex() const {return GetBindex();} //for the consistency with V0
  void     SetDcaXiDaughters(Double_t rDcaXiDaughters=0.);
  Double_t GetDcaXiDaughters() const {return fDcaXiDaughters;}
  Double_t GetCascadeCosineOfPointingAngle(Double_t refPointX, Double_t refPointY, Double_t refPointZ) const;
    
  void GetPosCovXi(Double_t cov[6]) const;

protected: 

  Double32_t fEffMassXi;      // reconstructed cascade effective mass
  Double32_t fChi2Xi;         // chi2 value
  Double32_t fDcaXiDaughters; // dca between Xi's daughters
  Double32_t fPosXi[3];       // cascade vertex position (global)
  Double32_t fPosCovXi[6];    // covariance matrix of the vertex position
  Double32_t fBachMom[3];     // bachelor momentum (global)
  Double32_t fBachMomCov[6];  // covariance matrix of the bachelor momentum.
  Int_t      fPdgCodeXi;      // reconstructed cascade type (PDG code)
  Int_t      fBachIdx;        // label of the bachelor track


private:


  ClassDef(AliESDcascade,6) // reconstructed cascade vertex
};

inline
void AliESDcascade::SetDcaXiDaughters(Double_t rDcaXiDaughters){
  fDcaXiDaughters=rDcaXiDaughters;
}

#endif
