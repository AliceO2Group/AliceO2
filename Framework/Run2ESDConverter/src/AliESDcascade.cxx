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

/* $Id$ */

//-------------------------------------------------------------------------
//             Implementation of the cascade vertex class
//              This is part of the Event Summary Data 
//              which contains the result of the reconstruction
//              and is the main set of classes for analaysis
//    Origin: Christian Kuhn, IReS, Strasbourg, christian.kuhn@ires.in2p3.fr
//     Modified by: Antonin Maire,IPHC, Antonin.Maire@ires.in2p3.fr
//             and Boris Hippolyte,IPHC, hippolyt@in2p3.fr
//          and David Chinellato, UNICAMP, daviddc@ifi.unicamp.br
//-------------------------------------------------------------------------

#include <TDatabasePDG.h>
#include <TMath.h>
#include <TVector3.h>
#include <TMatrixD.h>

#include "AliESDcascade.h"
#include "AliLog.h"

ClassImp(AliESDcascade)

AliESDcascade::AliESDcascade() : 
  AliESDv0(),
  fEffMassXi(TDatabasePDG::Instance()->GetParticle(kXiMinus)->Mass()),
  fChi2Xi(1024),
  fDcaXiDaughters(1024),
  fPdgCodeXi(kXiMinus),
  fBachIdx(-1)
{
  //--------------------------------------------------------------------
  // Default constructor  (Xi-)
  //--------------------------------------------------------------------
  for (Int_t j=0; j<3; j++) {
    fPosXi[j]=0.;
    fBachMom[j]=0.;
  }

  fPosCovXi[0]=1024;
  fPosCovXi[1]=fPosCovXi[2]=0.;
  fPosCovXi[3]=1024;
  fPosCovXi[4]=0.;
  fPosCovXi[5]=1024;

  fBachMomCov[0]=1024;
  fBachMomCov[1]=fBachMomCov[2]=0.;
  fBachMomCov[3]=1024;
  fBachMomCov[4]=0.;
  fBachMomCov[5]=1024;
}

AliESDcascade::AliESDcascade(const AliESDcascade& cas) :
  AliESDv0(cas),
  fEffMassXi(cas.fEffMassXi),
  fChi2Xi(cas.fChi2Xi),
  fDcaXiDaughters(cas.fDcaXiDaughters),
  fPdgCodeXi(cas.fPdgCodeXi),
  fBachIdx(cas.fBachIdx)
{
  //--------------------------------------------------------------------
  // The copy constructor
  //--------------------------------------------------------------------
  for (int i=0; i<3; i++) {
    fPosXi[i]     = cas.fPosXi[i];
    fBachMom[i] = cas.fBachMom[i];
  }
  for (int i=0; i<6; i++) {
    fPosCovXi[i]   = cas.fPosCovXi[i];
    fBachMomCov[i] = cas.fBachMomCov[i];
  }
}

AliESDcascade::AliESDcascade(const AliESDv0 &v,
			     const AliExternalTrackParam &t, Int_t i) : 
  AliESDv0(v),
  fEffMassXi(TDatabasePDG::Instance()->GetParticle(kXiMinus)->Mass()),
  fChi2Xi(1024),
  fDcaXiDaughters(1024),
  fPdgCodeXi(kXiMinus),
  fBachIdx(i)
{
  //--------------------------------------------------------------------
  // Main constructor  (Xi-)
  //--------------------------------------------------------------------

  Double_t r[3]; t.GetXYZ(r);
  Double_t x1=r[0], y1=r[1], z1=r[2]; // position of the bachelor
  Double_t p[3]; t.GetPxPyPz(p);
  Double_t px1=p[0], py1=p[1], pz1=p[2];// momentum of the bachelor track

  Double_t x2,y2,z2;          // position of the V0 
  v.GetXYZ(x2,y2,z2);    
  Double_t px2,py2,pz2;       // momentum of V0
  v.GetPxPyPz(px2,py2,pz2);

  Double_t a2=((x1-x2)*px2+(y1-y2)*py2+(z1-z2)*pz2)/(px2*px2+py2*py2+pz2*pz2);

  Double_t xm=x2+a2*px2;
  Double_t ym=y2+a2*py2;
  Double_t zm=z2+a2*pz2;

  // position of the cascade decay
  
  fPosXi[0]=0.5*(x1+xm); fPosXi[1]=0.5*(y1+ym); fPosXi[2]=0.5*(z1+zm);
    

  // invariant mass of the cascade (default is Ximinus)
  
  Double_t e1=TMath::Sqrt(0.13957*0.13957 + px1*px1 + py1*py1 + pz1*pz1);
  Double_t e2=TMath::Sqrt(1.11568*1.11568 + px2*px2 + py2*py2 + pz2*pz2);
  
  fEffMassXi=TMath::Sqrt((e1+e2)*(e1+e2)-
    (px1+px2)*(px1+px2)-(py1+py2)*(py1+py2)-(pz1+pz2)*(pz1+pz2));


  // momenta of the bachelor and the V0
  
  fBachMom[0]=px1; fBachMom[1]=py1; fBachMom[2]=pz1; 

  // Setting pdg code and fixing charge
  if (t.Charge()<0)
    fPdgCodeXi = kXiMinus;
  else
    fPdgCodeXi = kXiPlusBar;

  //PH Covariance matrices: to be calculated correctly in the future
  fPosCovXi[0]=1024;
  fPosCovXi[1]=fPosCovXi[2]=0.;
  fPosCovXi[3]=1024;
  fPosCovXi[4]=0.;
  fPosCovXi[5]=1024;

  fBachMomCov[0]=1024;
  fBachMomCov[1]=fBachMomCov[2]=0.;
  fBachMomCov[3]=1024;
  fBachMomCov[4]=0.;
  fBachMomCov[5]=1024;

  fChi2Xi=1024.; 

}

static Bool_t GetWeight(TMatrixD &w, const AliExternalTrackParam &t) {
  //
  // Returns the global weight matrix w = Transpose[G2P]*Inverse[Cpar]*G2P ,
  // where the matrix Cpar is the transverse part of the t covariance
  // in "parallel" system (i.e. the system with X axis parallel to momentum).
  // The matrix G2P performs the transformation global -> "parallel".
  //
  Double_t phi=t.GetAlpha() + TMath::ASin(t.GetSnp());
  Double_t sp=TMath::Sin(phi);
  Double_t cp=TMath::Cos(phi);
  
  Double_t tgl=t.GetTgl();
  Double_t cl=1/TMath::Sqrt(1.+ tgl*tgl);
  Double_t sl=tgl*cl;
  
  TMatrixD g2p(3,3); //global --> parallel
  g2p(0,0)= cp*cl; g2p(0,1)= sp*cl; g2p(0,2)=sl;
  g2p(1,0)=-sp;    g2p(1,1)= cp;    g2p(1,2)=0.;
  g2p(2,0)=-sl*cp; g2p(2,1)=-sl*sp; g2p(2,2)=cl;
  
  Double_t alpha=t.GetAlpha();
  Double_t c=TMath::Cos(alpha), s=TMath::Sin(alpha);
  TMatrixD l2g(3,3); //local --> global
  l2g(0,0)= c; l2g(0,1)=-s; l2g(0,2)= 0;
  l2g(1,0)= s; l2g(1,1)= c; l2g(1,2)= 0;
  l2g(2,0)= 0; l2g(2,1)= 0; l2g(2,2)= 1;
  
  Double_t sy2=t.GetSigmaY2(), syz=t.GetSigmaZY(), sz2=t.GetSigmaZ2();
  TMatrixD cvl(3,3); //local covariance
  cvl(0,0)=0; cvl(0,1)=0;   cvl(0,2)=0;
  cvl(1,0)=0; cvl(1,1)=sy2; cvl(1,2)=syz;
  cvl(2,0)=0; cvl(2,1)=syz; cvl(2,2)=sz2;
  
  TMatrixD l2p(g2p, TMatrixD::kMult, l2g);
  TMatrixD cvp(3,3); //parallel covariance
  cvp=l2p*cvl*TMatrixD(TMatrixD::kTransposed,l2p);
  
  Double_t det=cvp(1,1)*cvp(2,2) - cvp(1,2)*cvp(2,1);
  if (TMath::Abs(det)<kAlmost0) return kFALSE;
  
  const Double_t m=100*100; //A large uncertainty in the momentum direction
  const Double_t eps=1/m;
  TMatrixD u(3,3);  //Inverse of the transverse part of the parallel covariance
  u(0,0)=eps; u(0,1)=0;              u(0,2)=0;
  u(1,0)=0;   u(1,1)= cvp(2,2)/det;  u(1,2)=-cvp(2,1)/det;
  u(2,0)=0;   u(2,1)=-cvp(1,2)/det;  u(2,2)= cvp(1,1)/det;
  
  w=TMatrixD(TMatrixD::kTransposed,g2p)*u*g2p;
  
  return kTRUE;
}

Int_t AliESDcascade::RefitCascade(AliExternalTrackParam *bachelor)
{
  //--------------------------------------------------------------------
  // Refit cascade decay vertex
  //--------------------------------------------------------------------
  //
  // CAUTION: Requires AliESDv0::Refit() to have been called, since
  // covariance matrix of the v0 decay vertex will be used!
  //
  // CAUTION: Requires a bachelor track already located at the
  // expected point of closest approach to the V0!
  //
  // Mathematical procedure for vertex position recalculation:
  //
  // r = [(C1^-1) + (C2^-1)]^-1 * [ (C1^-1)*r1 + (C2^-1)*r2 ]
  //
  //    r:  uncertainty-aware vertex position
  //    r1: V0 position of DCA (note: NOT DECAY POSITION)
  //    r2: bachelor position of DCA
  //    C1: covariance matrix of V0 decay position
  //    C2: covariance matrix of bachelor position
  //     (here, "^-1" means matrix inversion!)
  //
  // N.B.: This procedure approximates the V0 uncertainty at the point
  // of DCA to the bachelor to be approximately the same as the
  // one measured at the V0 decay vertex. This is reasonable at low
  // pT but may not be ideal at high pT, where, however, resolution
  // is typically much better.
  //
  // In addition to repositioning the cascade at a better decay
  // position, this procedure also fills out the cascade chi2 value
  // and the cascade decay position covariance matrix, which may then
  // be used.
  //
  // Note about arguments: the negative and positive tracks of the V0
  // are contained as data members of the cascade class, but the
  // bachelor track isn't, which makes it necessary to pass a pointer
  // to that...
  //
  //--------------------------------------------------------------------
  
  //____________________________________________________________
  //Step 1: Acquire positions r1, r2
  
  //Bachelor position: r2
  Double_t r2[3];
  bachelor->GetXYZ(r2);

  //WARNING: V0 has to be back-propagated!
  //V0 characteristics
  //Position
  Double_t xv,yv,zv;
  xv = fPos[0]; yv = fPos[1]; zv = fPos[2];
  
  //Momentum
  Double_t pxv,pyv,pzv;
  pxv = fNmom[0]+fPmom[0];
  pyv = fNmom[1]+fPmom[1];
  pzv = fNmom[2]+fPmom[2];

  Double_t a2=((r2[0]-xv)*pxv+(r2[1]-yv)*pyv+(r2[2]-zv)*pzv)/(pxv*pxv+pyv*pyv+pzv*pzv);
  
  //V0 position: r1
  Double_t r1[3];
  r1[0]=xv+a2*pxv;
  r1[1]=yv+a2*pyv;
  r1[2]=zv+a2*pzv;
  
  //____________________________________________________________
  //Step 2: Acquire cov matrices C1, C2
  TMatrixD C1(3,3); //V0 cov mat
  C1(0,0) = fPosCov[0]; C1(0,1) = fPosCov[1]; C1(0,2) = fPosCov[3];
  C1(1,0) = fPosCov[1]; C1(1,1) = fPosCov[2]; C1(1,2) = fPosCov[4];
  C1(2,0) = fPosCov[3]; C1(2,1) = fPosCov[4]; C1(2,2) = fPosCov[5];
  
  C1.Invert();
  if( !C1.IsValid() ) return kFALSE;
  
  TMatrixD C2(3,3); //bach cov mat
  AliExternalTrackParam lBach(*bachelor);
  if( !GetWeight(C2,lBach) ) return kFALSE;
  
  //____________________________________________________________
  //Step 3: Calculate covariance of cascade vertex position
  TMatrixD lPosCovXi(C1); lPosCovXi+=C2;
  lPosCovXi.Invert();
  if (!lPosCovXi.IsValid()) return kFALSE;
  
  //Covariance of the V0 vertex
  fPosCovXi[0]=lPosCovXi(0,0);
  fPosCovXi[1]=lPosCovXi(1,0); fPosCovXi[2]=lPosCovXi(1,1);
  fPosCovXi[3]=lPosCovXi(2,0); fPosCovXi[4]=lPosCovXi(2,1); fPosCovXi[5]=lPosCovXi(2,2);
  
  //____________________________________________________________
  //Step 4: Calculate cascade vertex position
  Double_t lVec[3]; //helper vector
  
  lVec[0]  = C1(0,0)*r1[0] + C1(0,1)*r1[1] + C1(0,2)*r1[2];
  lVec[1]  = C1(1,0)*r1[0] + C1(1,1)*r1[1] + C1(1,2)*r1[2];
  lVec[2]  = C1(2,0)*r1[0] + C1(2,1)*r1[1] + C1(2,2)*r1[2];
  
  lVec[0] += C2(0,0)*r2[0] + C2(0,1)*r2[1] + C2(0,2)*r2[2];
  lVec[1] += C2(1,0)*r2[0] + C2(1,1)*r2[1] + C2(1,2)*r2[2];
  lVec[2] += C2(2,0)*r2[0] + C2(2,1)*r2[1] + C2(2,2)*r2[2];
  
  fPosXi[0] = lPosCovXi(0,0)*lVec[0] + lPosCovXi(0,1)*lVec[1] + lPosCovXi(0,2)*lVec[2];
  fPosXi[1] = lPosCovXi(1,0)*lVec[0] + lPosCovXi(1,1)*lVec[1] + lPosCovXi(1,2)*lVec[2];
  fPosXi[2] = lPosCovXi(2,0)*lVec[0] + lPosCovXi(2,1)*lVec[1] + lPosCovXi(2,2)*lVec[2];
  
  //____________________________________________________________
  //Step 5: Calculate cascade chi2
  fChi2Xi = 0.0;
  Double_t res1[3]={r1[0]-fPosXi[0],r1[1]-fPosXi[1],r1[2]-fPosXi[2]};
  Double_t res2[3]={r2[0]-fPosXi[0],r2[1]-fPosXi[1],r2[2]-fPosXi[2]};
  for (Int_t i=0; i<3; i++)
    for (Int_t j=0; j<3; j++)
      fChi2Xi += res1[i]*res1[j]*C1(i,j) + res2[i]*res2[j]*C2(i,j);
  
  return kTRUE; //done and OK!
}

AliESDcascade& AliESDcascade::operator=(const AliESDcascade& cas)
{
  //--------------------------------------------------------------------
  // The assignment operator
  //--------------------------------------------------------------------

  if(this==&cas) return *this;
  AliESDv0::operator=(cas);

  fEffMassXi = cas.fEffMassXi;
  fChi2Xi  = cas.fChi2Xi;
  fDcaXiDaughters = cas.fDcaXiDaughters;
  fPdgCodeXi      = cas.fPdgCodeXi;
  fBachIdx        = cas.fBachIdx;
  for (int i=0; i<3; i++) {
    fPosXi[i]     = cas.fPosXi[i];
    fBachMom[i]   = cas.fBachMom[i];
  }
  for (int i=0; i<6; i++) {
    fPosCovXi[i]   = cas.fPosCovXi[i];
    fBachMomCov[i] = cas.fBachMomCov[i];
  }
  return *this;
}

void AliESDcascade::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDcascade *robj = dynamic_cast<AliESDcascade*>(&obj);
  if(!robj)return; // not an AliESDcascade
  *robj = *this;
}

AliESDcascade::~AliESDcascade() {
  //--------------------------------------------------------------------
  // Empty destructor
  //--------------------------------------------------------------------
}

// Start with AliVParticle functions
Double_t AliESDcascade::E() const {
  //--------------------------------------------------------------------
  // This gives the energy assuming the ChangeMassHypothesis was called
  //--------------------------------------------------------------------
  return E(fPdgCodeXi);
}

Double_t AliESDcascade::Y() const {
  //--------------------------------------------------------------------
  // This gives the energy assuming the ChangeMassHypothesis was called
  //--------------------------------------------------------------------
  return Y(fPdgCodeXi);
}

// Then extend AliVParticle functions
Double_t AliESDcascade::E(Int_t pdg) const {
  //--------------------------------------------------------------------
  // This gives the energy with the particle hypothesis as argument 
  //--------------------------------------------------------------------
  Double_t mass = TDatabasePDG::Instance()->GetParticle(pdg)->Mass();
  return TMath::Sqrt(mass*mass+P()*P());
}

Double_t AliESDcascade::Y(Int_t pdg) const {
  //--------------------------------------------------------------------
  // This gives the rapidity with the particle hypothesis as argument 
  //--------------------------------------------------------------------
  return 0.5*TMath::Log((E(pdg)+Pz())/(E(pdg)-Pz()+1.e-13));
}

// Now the functions for analysis consistency
Double_t AliESDcascade::RapXi() const {
  //--------------------------------------------------------------------
  // This gives the pseudorapidity assuming a (Anti) Xi particle
  //--------------------------------------------------------------------
  return Y(kXiMinus);
}

Double_t AliESDcascade::RapOmega() const {
  //--------------------------------------------------------------------
  // This gives the pseudorapidity assuming a (Anti) Omega particle
  //--------------------------------------------------------------------
  return Y(kOmegaMinus);
}

Double_t AliESDcascade::AlphaXi() const {
  //--------------------------------------------------------------------
  // This gives the Armenteros-Podolanski alpha
  //--------------------------------------------------------------------
  TVector3 momBach(fBachMom[0],fBachMom[1],fBachMom[2]);
  TVector3 momV0(fNmom[0]+fPmom[0],fNmom[1]+fPmom[1],fNmom[2]+fPmom[2]);
  TVector3 momTot(Px(),Py(),Pz());

  Double_t lQlBach = momBach.Dot(momTot)/momTot.Mag();
  Double_t lQlV0 = momV0.Dot(momTot)/momTot.Mag();

  return 1.-2./(1.+lQlBach/lQlV0);
}

Double_t AliESDcascade::PtArmXi() const {
  //--------------------------------------------------------------------
  // This gives the Armenteros-Podolanski ptarm
  //--------------------------------------------------------------------
  TVector3 momBach(fBachMom[0],fBachMom[1],fBachMom[2]);
  TVector3 momTot(Px(),Py(),Pz());

  return momBach.Perp(momTot);
}

// Then the older functions
Double_t AliESDcascade::ChangeMassHypothesis(Double_t &v0q, Int_t code) {
  //--------------------------------------------------------------------
  // This function changes the mass hypothesis for this cascade
  // and returns the "kinematical quality" of this hypothesis
  // together with the "quality" of associated V0 (argument v0q) 
  //--------------------------------------------------------------------
  Double_t nmass=0.13957, pmass=0.93827, ps0=0.101; 
  Double_t bmass=0.13957, mass =1.3213,  ps =0.139;

  if (Charge()*code<0)
    fPdgCodeXi = code;
  else {
    AliWarning("Chosen PDG code does not match the sign of the bachelor... Corrected !!");
    fPdgCodeXi = -code;
  }

  switch (fPdgCodeXi) {
  case kXiMinus:
       break;
  case kXiPlusBar:
       nmass=0.93827; pmass=0.13957; 
       break;
  case kOmegaMinus: 
       bmass=0.49368; mass=1.67245; ps=0.211;
       break;
  case kOmegaPlusBar: 
       nmass=0.93827; pmass=0.13957; 
       bmass=0.49368; mass=1.67245; ps=0.211;
       break;
  default:
       AliError("Invalide PDG code !  Assuming a Xi particle...");
       if (Charge()<0) {
	 fPdgCodeXi=kXiMinus;
       }
       else {
	 fPdgCodeXi=kXiPlusBar;
	 nmass=0.93827; pmass=0.13957; 
       }
    break;
  }

  Double_t pxn=fNmom[0], pyn=fNmom[1], pzn=fNmom[2];
  Double_t pxp=fPmom[0], pyp=fPmom[1], pzp=fPmom[2];

  Double_t px0=pxn+pxp, py0=pyn+pyp, pz0=pzn+pzp;
  Double_t p0=TMath::Sqrt(px0*px0 + py0*py0 + pz0*pz0);

  Double_t e0=TMath::Sqrt(1.11568*1.11568 + p0*p0);
  Double_t beta0=p0/e0;
  Double_t pln=(pxn*px0 + pyn*py0 + pzn*pz0)/p0;
  Double_t plp=(pxp*px0 + pyp*py0 + pzp*pz0)/p0;
  Double_t pt2=pxp*pxp + pyp*pyp + pzp*pzp - plp*plp;

  Double_t a=(plp-pln)/(plp+pln);
  a -= (pmass*pmass-nmass*nmass)/(1.11568*1.11568);
  a = 0.25*beta0*beta0*1.11568*1.11568*a*a + pt2;


  v0q=a - ps0*ps0;


  Double_t pxb=fBachMom[0], pyb=fBachMom[1], pzb=fBachMom[2]; 

  Double_t eb=TMath::Sqrt(bmass*bmass + pxb*pxb + pyb*pyb + pzb*pzb);
  Double_t pxl=px0+pxb, pyl=py0+pyb, pzl=pz0+pzb;
  Double_t pl=TMath::Sqrt(pxl*pxl + pyl*pyl + pzl*pzl);
  
  fEffMassXi=TMath::Sqrt(((e0+eb)-pl)*((e0+eb)+pl));

  Double_t beta=pl/(e0+eb);
  Double_t pl0=(px0*pxl + py0*pyl + pz0*pzl)/pl;
  Double_t plb=(pxb*pxl + pyb*pyl + pzb*pzl)/pl;
  pt2=p0*p0 - pl0*pl0;

  a=(pl0-plb)/(pl0+plb);
  a -= (1.11568*1.11568-bmass*bmass)/(mass*mass);
  a = 0.25*beta*beta*mass*mass*a*a + pt2;

  return (a - ps*ps);
}

void 
AliESDcascade::GetPxPyPz(Double_t &px, Double_t &py, Double_t &pz) const {
  //--------------------------------------------------------------------
  // This function returns the cascade momentum (global)
  //--------------------------------------------------------------------
  px=fNmom[0]+fPmom[0]+fBachMom[0];
  py=fNmom[1]+fPmom[1]+fBachMom[1]; 
  pz=fNmom[2]+fPmom[2]+fBachMom[2]; 
}

void AliESDcascade::GetXYZcascade(Double_t &x, Double_t &y, Double_t &z) const {
  //--------------------------------------------------------------------
  // This function returns cascade position (global)
  //--------------------------------------------------------------------
  x=fPosXi[0];
  y=fPosXi[1];
  z=fPosXi[2];
}

void AliESDcascade::SetXYZcascade(Double_t x, Double_t y, Double_t z) {
    //--------------------------------------------------------------------
    // This function sets cascade position (global)
    //--------------------------------------------------------------------
    fPosXi[0]=x;
    fPosXi[1]=y;
    fPosXi[2]=z;
}

Double_t AliESDcascade::GetDcascade(Double_t x0, Double_t y0, Double_t z0) const {
  //--------------------------------------------------------------------
  // This function returns the cascade impact parameter
  //--------------------------------------------------------------------

  Double_t x=fPosXi[0],y=fPosXi[1],z=fPosXi[2];
  Double_t px=fNmom[0]+fPmom[0]+fBachMom[0];
  Double_t py=fNmom[1]+fPmom[1]+fBachMom[1];
  Double_t pz=fNmom[2]+fPmom[2]+fBachMom[2];

  Double_t dx=(y0-y)*pz - (z0-z)*py; 
  Double_t dy=(x0-x)*pz - (z0-z)*px;
  Double_t dz=(x0-x)*py - (y0-y)*px;
  Double_t d=TMath::Sqrt((dx*dx+dy*dy+dz*dz)/(px*px+py*py+pz*pz));

  return d;
}

Double_t AliESDcascade::GetCascadeCosineOfPointingAngle(Double_t refPointX, Double_t refPointY, Double_t refPointZ) const {
  // calculates the pointing angle of the cascade wrt a reference point

  Double_t momCas[3]; //momentum of the cascade
  GetPxPyPz(momCas[0],momCas[1],momCas[2]);

  Double_t deltaPos[3]; //vector between the reference point and the cascade vertex
  deltaPos[0] = fPosXi[0] - refPointX;
  deltaPos[1] = fPosXi[1] - refPointY;
  deltaPos[2] = fPosXi[2] - refPointZ;

  Double_t momCas2    = momCas[0]*momCas[0] + momCas[1]*momCas[1] + momCas[2]*momCas[2];
  Double_t deltaPos2 = deltaPos[0]*deltaPos[0] + deltaPos[1]*deltaPos[1] + deltaPos[2]*deltaPos[2];

  Double_t cosinePointingAngle = (deltaPos[0]*momCas[0] +
				  deltaPos[1]*momCas[1] +
				  deltaPos[2]*momCas[2] ) /
    TMath::Sqrt(momCas2 * deltaPos2);
  
  return cosinePointingAngle;
}

void AliESDcascade::GetPosCovXi(Double_t cov[6]) const {

  for (Int_t i=0; i<6; ++i) cov[i] = fPosCovXi[i];
}
