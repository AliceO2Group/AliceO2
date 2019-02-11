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

//***********************************************************
// Class AliHMPIDPIDResponse
//
// HMPID class to perfom particle identification
// 
// Author: G. Volpe, giacomo.volpe@cern.ch
//***********************************************************


#include "AliHMPIDPIDResponse.h"  //class header
#include "AliPID.h"               //FindPid()
#include "AliVTrack.h"            //FindPid()
#include "AliLog.h"               //general
#include <TRandom.h>              //Resolution()
#include <TVector2.h>             //Resolution()
#include <TRotation.h>
#include <TF1.h>
#include <TGeoManager.h>          //Instance()
#include <TGeoMatrix.h>           //Instance()
#include <TGeoPhysicalNode.h>     //ctor
#include <TGeoBBox.h>
#include <TObjArray.h>

Float_t AliHMPIDPIDResponse::fgkMinPcX[]={0.,0.,0.,0.,0.,0.};
Float_t AliHMPIDPIDResponse::fgkMaxPcX[]={0.,0.,0.,0.,0.,0.};
Float_t AliHMPIDPIDResponse::fgkMinPcY[]={0.,0.,0.,0.,0.,0.};
Float_t AliHMPIDPIDResponse::fgkMaxPcY[]={0.,0.,0.,0.,0.,0.};

Float_t AliHMPIDPIDResponse::fgCellX=0.;
Float_t AliHMPIDPIDResponse::fgCellY=0.;

Float_t AliHMPIDPIDResponse::fgPcX=0;
Float_t AliHMPIDPIDResponse::fgPcY=0;

Float_t AliHMPIDPIDResponse::fgAllX=0;
Float_t AliHMPIDPIDResponse::fgAllY=0;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AliHMPIDPIDResponse::AliHMPIDPIDResponse():
  TNamed("HMPIDPIDResponseRec","HMPIDPIDResponsePid"),    
  fRefIdx(1.28947),
  fTrkDir(0,0,1),  
  fTrkPos(30,40),  
  fRefIndexArray(0x0)
{
  //
  // ctor
  //
  
  Float_t dead=2.6;// cm of the dead zones between PCs-> See 2CRC2099P1

  fgCellX=0.8; fgCellY=0.84;
  
  fgPcX  = 80.*fgCellX;      fgPcY = 48.*fgCellY;
  fgAllX = 2.*fgPcX+dead;
  fgAllY = 3.*fgPcY+2.*dead;

  fgkMinPcX[1]=fgPcX+dead; fgkMinPcX[3]=fgkMinPcX[1];  fgkMinPcX[5]=fgkMinPcX[3];
  fgkMaxPcX[0]=fgPcX;      fgkMaxPcX[2]=fgkMaxPcX[0];  fgkMaxPcX[4]=fgkMaxPcX[2];
  fgkMaxPcX[1]=fgAllX;     fgkMaxPcX[3]=fgkMaxPcX[1];  fgkMaxPcX[5]=fgkMaxPcX[3];

  fgkMinPcY[2]=fgPcY+dead;       fgkMinPcY[3]=fgkMinPcY[2];  
  fgkMinPcY[4]=2.*fgPcY+2.*dead; fgkMinPcY[5]=fgkMinPcY[4];
  fgkMaxPcY[0]=fgPcY;            fgkMaxPcY[1]=fgkMaxPcY[0];  
  fgkMaxPcY[2]=2.*fgPcY+dead;    fgkMaxPcY[3]=fgkMaxPcY[2]; 
  fgkMaxPcY[4]=fgAllY;           fgkMaxPcY[5]=fgkMaxPcY[4];   
    
  for(Int_t i=kMinCh;i<=kMaxCh;i++)
    if(gGeoManager && gGeoManager->IsClosed()) {
      TGeoPNEntry* pne = gGeoManager->GetAlignableEntry(Form("/HMPID/Chamber%i",i));
      if (!pne) {
        AliErrorClass(Form("The symbolic volume %s does not correspond to any physical entry!",Form("HMPID_%i",i)));
        fM[i]=new TGeoHMatrix;
        IdealPosition(i,fM[i]);
      } else {
        TGeoPhysicalNode *pnode = pne->GetPhysicalNode();
        if(pnode) fM[i]=new TGeoHMatrix(*(pnode->GetMatrix()));
        else {
          fM[i]=new TGeoHMatrix;
          IdealPosition(i,fM[i]);
        }
      }
    } else{
      fM[i]=new TGeoHMatrix;
      IdealPosition(i,fM[i]);
    } 
    
}//ctor
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AliHMPIDPIDResponse::AliHMPIDPIDResponse(const AliHMPIDPIDResponse& c):
  TNamed(c), 
  fRefIdx(c.fRefIdx),
  fTrkDir(c.fTrkDir),  
  fTrkPos(c.fTrkPos),  
  fRefIndexArray(c.fRefIndexArray)
 {
   //
   // copy ctor
   //
   
   for(Int_t i=0; i<6; i++) {
      
      fgkMinPcX[i] = c.fgkMinPcX[i];
      fgkMinPcY[i] = c.fgkMinPcY[i];
      fgkMaxPcX[i] = c.fgkMaxPcX[i];
      fgkMaxPcY[i] = c.fgkMaxPcY[i];
     }
   
   for(Int_t i=0; i<7; i++) fM[i] = c.fM[i] ? new TGeoHMatrix(*c.fM[i]) : 0;
 }
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AliHMPIDPIDResponse::~AliHMPIDPIDResponse()
{
  // d-tor
  for (int i=7;i--;) delete fM[i];
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
AliHMPIDPIDResponse& AliHMPIDPIDResponse::operator=(const AliHMPIDPIDResponse& c) {

   //
   // assignment operator
   //       
  if(this!=&c){
     TNamed::operator=(c);
     fgCellX = c.fgCellX;
     fgCellY = c.fgCellY;
     fgPcX   = c.fgPcX;
     fgPcY   = c.fgPcY;
     fgAllX  = c.fgAllX;
     fgAllY  = c.fgAllY;
     fRefIdx = c.fRefIdx;
     fTrkDir = c.fTrkDir;  
     fTrkPos = c.fTrkPos;  
     fRefIndexArray = c.fRefIndexArray;
     for(Int_t i=0; i<6; i++) {    
      fgkMinPcX[i] = c.fgkMinPcX[i];
      fgkMinPcY[i] = c.fgkMinPcY[i];
      fgkMaxPcX[i] = c.fgkMaxPcX[i];
      fgkMaxPcY[i] = c.fgkMaxPcY[i];
     }   
     for(Int_t i=0; i<7; i++) fM[i] = c.fM[i] ? new TGeoHMatrix(*c.fM[i]) : 0;
    } 
    
  return *this; 
}    
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void AliHMPIDPIDResponse::IdealPosition(Int_t iCh, TGeoHMatrix *pMatrix) {

// Construct ideal position matrix for a given chamber
// Arguments: iCh- chamber ID; pMatrix- pointer to precreated unity matrix where to store the results
// Returns: none

  const Double_t kAngHor=19.5;        //  horizontal angle between chambers  19.5 grad
  const Double_t kAngVer=20;          //  vertical angle between chambers    20   grad     
  const Double_t kAngCom=30;          //  common HMPID rotation with respect to x axis  30   grad     
  const Double_t kTrans[3]={490,0,0}; //  center of the chamber is on window-gap surface
  pMatrix->RotateY(90);               //  rotate around y since initial position is in XY plane -> now in YZ plane
  pMatrix->SetTranslation(kTrans);    //  now plane in YZ is shifted along x 
  switch(iCh){
    case 0:                pMatrix->RotateY(kAngHor);  pMatrix->RotateZ(-kAngVer);  break; //right and down 
    case 1:                                            pMatrix->RotateZ(-kAngVer);  break; //down              
    case 2:                pMatrix->RotateY(kAngHor);                               break; //right 
    case 3:                                                                         break; //no rotation
    case 4:                pMatrix->RotateY(-kAngHor);                              break; //left   
    case 5:                                            pMatrix->RotateZ(kAngVer);   break; //up
    case 6:                pMatrix->RotateY(-kAngHor); pMatrix->RotateZ(kAngVer);   break; //left and up 
  }
  pMatrix->RotateZ(kAngCom);     //apply common rotation  in XY plane    
   
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::GetExpectedSignal(const AliVTrack *vTrk, AliPID::EParticleType specie) const {
  
  // expected Cherenkov angle calculation
  
  const Double_t nmean = GetNMean(vTrk);
  return ExpectedSignal(vTrk,nmean,specie);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::GetExpectedSigma(const AliVTrack *vTrk, AliPID::EParticleType specie) const {
  
  // expected resolution calculation
  
  const Double_t nmean = GetNMean(vTrk);
  return ExpectedSigma(vTrk,nmean,specie);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::ExpectedSignal(const AliVTrack *vTrk, Double_t nmean, AliPID::EParticleType specie) const {
  
  // expected Cherenkov angle calculation
  
  Double_t thetaTheor = -999.;
  
  Double_t p[3] = {0}, mom = 0;
  if(vTrk->GetOuterHmpPxPyPz(p))  mom = TMath::Sqrt(p[0]*p[0]+p[1]*p[1]+p[2]*p[2]);  // Momentum of the charged particle    
  else return thetaTheor;
    
  if(mom<0.001) return thetaTheor;
                  
  const Double_t mass = AliPID::ParticleMass(specie); 
  const Double_t cosTheta = TMath::Sqrt(mass*mass+mom*mom)/(nmean*mom);
   
  if(cosTheta>1) return thetaTheor;
                  
  else thetaTheor = TMath::ACos(cosTheta);
  
  return thetaTheor;                                                                                          //  evaluate the theor. Theta Cherenkov
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::ExpectedSigma(const AliVTrack *vTrk, Double_t nmean, AliPID::EParticleType specie) const {
  
  // expected resolution calculation
  
  Float_t x=0., y=0.;
  Int_t q=0, nph=0;
  Float_t xPc=0.,yPc=0.,thRa=0.,phRa=0.;
  
  vTrk->GetHMPIDmip(x,y,q,nph);  
  vTrk->GetHMPIDtrk(xPc,yPc,thRa,phRa);
      
  const Double_t xRa = xPc - (RadThick()+WinThick()+GapThick())*TMath::Cos(phRa)*TMath::Tan(thRa);  //just linear extrapolation back to RAD
  const Double_t yRa = yPc - (RadThick()+WinThick()+GapThick())*TMath::Sin(phRa)*TMath::Tan(thRa);  //just linear extrapolation back to RAD
  
  const Double_t thetaCerTh = ExpectedSignal(vTrk,nmean,specie);
  const Double_t occupancy  = vTrk->GetHMPIDoccupancy();
  const Double_t thetaMax   = TMath::ACos(1./nmean);
  const Int_t    nPhotsTh   = (Int_t)(12.*TMath::Sin(thetaCerTh)*TMath::Sin(thetaCerTh)/(TMath::Sin(thetaMax)*TMath::Sin(thetaMax))+0.01);

  Double_t sigmatot = 0;
  Int_t nTrks = 20;
  for(Int_t iTrk=0;iTrk<nTrks;iTrk++) {
    Double_t invSigma = 0;
    Int_t nPhotsAcc = 0;
    
    Int_t nPhots = 0; 
    if(nph<nPhotsTh+TMath::Sqrt(nPhotsTh) && nph>nPhotsTh-TMath::Sqrt(nPhotsTh)) nPhots = nph;
    else nPhots = gRandom->Poisson(nPhotsTh);
    
    for(Int_t j=0;j<nPhots;j++){
      Double_t phi = gRandom->Rndm()*TMath::TwoPi();
      TVector2 pos; pos = TracePhot(xRa,yRa,thRa,phRa,thetaCerTh,phi);
      if(!IsInside(pos.X(),pos.Y())) continue;
      if(IsInDead(pos.X(),pos.Y()))  continue;
      Double_t sigma2 = Sigma2(thRa,phRa,thetaCerTh,phi); //photon candidate sigma^2
      
      if(sigma2!=0) {
        invSigma += 1./sigma2;
        nPhotsAcc++;
      }
    }      
    if(invSigma!=0) sigmatot += 1./TMath::Sqrt(invSigma);  
  }
    
  return (sigmatot/nTrks)*SigmaCorrFact(specie,occupancy);
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::GetNumberOfSigmas(const AliVTrack *vTrk, AliPID::EParticleType specie) const {

  // Number of sigmas calculation
    
  Double_t nSigmas = -999.;

  if(vTrk->GetHMPIDsignal()<0.) return nSigmas;
    
  const Double_t nmean = GetNMean(vTrk);
  
  const Double_t expSigma = ExpectedSigma(vTrk, nmean, specie);
  
  if(expSigma > 0.) nSigmas = (vTrk->GetHMPIDsignal() - ExpectedSignal(vTrk,nmean,specie))/expSigma;
                         
  return nSigmas;
    
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void AliHMPIDPIDResponse::GetProbability(const AliVTrack *vTrk,Int_t nSpecies,Double_t *prob) const {

// Calculates probability to be a electron-muon-pion-kaon-proton with the "amplitude" method
// from the given Cerenkov angle and momentum assuming no initial particle composition
  
  const Double_t thetaCerExp = vTrk->GetHMPIDsignal();                                                                           

  const Double_t nmean = GetNMean(vTrk);
    
  if(thetaCerExp<=0){                                                                                     // HMPID does not find anything reasonable for this track, assign 0.2 for all species
    for(Int_t iPart=0;iPart<nSpecies;iPart++) prob[iPart]=1.0/(Float_t)nSpecies;
    return;
  } 
  
  Double_t p[3] = {0,0,0};
  
  if(!(vTrk->GetOuterHmpPxPyPz(p))) for(Int_t iPart=0;iPart<nSpecies;iPart++) prob[iPart]=1.0/(Float_t)nSpecies;
  
  Double_t hTot=0;                                                                                        // Initialize the total height of the amplitude method
  Double_t *h = new Double_t [nSpecies];                                                                  // number of charged particles to be considered

  Bool_t desert = kTRUE;                                                                                  // Flag to evaluate if ThetaC is far ("desert") from the given Gaussians
  
  for(Int_t iPart=0;iPart<nSpecies;iPart++){                                                              // for each particle

        
    h[iPart] = 0;                                                                                         // reset the height
    Double_t thetaCerTh = ExpectedSignal(vTrk,nmean,(AliPID::EParticleType)iPart);                        // theoretical Theta Cherenkov
    if(thetaCerTh>900.) continue;                                                                         // no light emitted, zero height
    Double_t sigmaRing = ExpectedSigma(vTrk,nmean,(AliPID::EParticleType)iPart);
    
    if(sigmaRing==0) continue;
      
    if(TMath::Abs(thetaCerExp-thetaCerTh)<4*sigmaRing) desert = kFALSE;                                                               
    h[iPart] =TMath::Gaus(thetaCerTh,thetaCerExp,sigmaRing,kTRUE);
    hTot    +=h[iPart];                                                                                   // total height of all theoretical heights for normalization
    
  }//species loop

  for(Int_t iPart=0;iPart<nSpecies;iPart++) {                                                             // species loop to assign probabilities
     
    if(!desert) prob[iPart]=h[iPart]/hTot;
    else        prob[iPart]=1.0/(Float_t)nSpecies;                                                        // all theoretical values are far away from experemental one
    
  }
  
  delete [] h;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::GetSignalDelta(const AliVTrack *vTrk, AliPID::EParticleType specie, Bool_t ratio/*=kFALSE*/) const {
  
  //
  // calculation of Experimental Cherenkov angle - Theoretical Cherenkov angle  
  //
  const Double_t signal    = vTrk->GetHMPIDsignal();
  const Double_t expSignal = GetExpectedSignal(vTrk,specie);

  Double_t delta = -9999.;
  if (!ratio) delta=signal-expSignal;
  else if (expSignal>1.e-20) delta=signal/expSignal;
  
  return delta;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TVector2 AliHMPIDPIDResponse::TracePhot(Double_t xRa, Double_t yRa, Double_t thRa, Double_t phRa, Double_t ckovThe,Double_t ckovPhi) const {

// Trace a single Ckov photon from emission point somewhere in radiator up to photocathode taking into account ref indexes of materials it travereses
// Returns: distance between photon point on PC and track projection  
  
  Double_t theta=0.,phi=0.;
  TVector3  dirTRS,dirLORS;
  dirTRS.SetMagThetaPhi(1,ckovThe,ckovPhi);                     //photon in TRS
  Trs2Lors(thRa,phRa,dirTRS,theta,phi);
  dirLORS.SetMagThetaPhi(1,theta,phi);                          //photon in LORS
  return TraceForward(xRa,yRa,dirLORS);                                 //now foward tracing
}//TracePhot()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
TVector2 AliHMPIDPIDResponse::TraceForward(Double_t xRa, Double_t yRa, TVector3 dirCkov) const {

// Trace forward a photon from (x,y) up to PC
// Returns: pos of traced photon at PC
  
  TVector2 pos(-999,-999);
  Double_t thetaCer = dirCkov.Theta();
  if(thetaCer > TMath::ASin(1./GetRefIdx())) return pos;          //total refraction on WIN-GAP boundary
  Double_t zRad= -0.5*RadThick()-0.5*WinThick();          //z position of middle of RAD
  TVector3  posCkov(xRa,yRa,zRad);                        //RAD: photon position is track position @ middle of RAD 
  Propagate(dirCkov,posCkov,           -0.5*WinThick());          //go to RAD-WIN boundary  
  Refract  (dirCkov,         GetRefIdx(),WinIdx());       //RAD-WIN refraction
  Propagate(dirCkov,posCkov,            0.5*WinThick());          //go to WIN-GAP boundary
  Refract  (dirCkov,         WinIdx(),GapIdx());          //WIN-GAP refraction
  Propagate(dirCkov,posCkov,0.5*WinThick()+GapThick());   //go to PC
  pos.Set(posCkov.X(),posCkov.Y());
  return pos;
}//TraceForward()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void AliHMPIDPIDResponse::Propagate(const TVector3 dir,TVector3 &pos,Double_t z) const {
  
// Finds an intersection point between a line and XY plane shifted along Z.
// Arguments:  dir,pos   - vector along the line and any point of the line
//             z         - z coordinate of plain 
// Returns:  none
// On exit:  pos is the position if this intesection if any

  static TVector3 nrm(0,0,1); 
         TVector3 pnt(0,0,z);
  
  TVector3 diff=pnt-pos;
  Double_t sint=(nrm*diff)/(nrm*dir);
  pos+=sint*dir;
}//Propagate()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void AliHMPIDPIDResponse::Refract(TVector3 &dir,Double_t n1,Double_t n2) const {

// Refract direction vector according to Snell law
// Arguments: 
//            n1 - ref idx of first substance
//            n2 - ref idx of second substance
//   Returns: none
//   On exit: dir is new direction
  
  Double_t sinref=(n1/n2)*TMath::Sin(dir.Theta());
  if(TMath::Abs(sinref)>1.) dir.SetXYZ(-999,-999,-999);
  else             dir.SetTheta(TMath::ASin(sinref));
}//Refract()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
void AliHMPIDPIDResponse::Trs2Lors(Double_t thRa, Double_t phRa, TVector3 dirCkov,Double_t &thetaCer,Double_t &phiCer) const {

  // Theta Cerenkov reconstruction 
  // Returns: thetaCer of photon in LORS
  //          phiCer of photon in LORS
  
  TRotation mtheta;   mtheta.RotateY(thRa);
  TRotation mphi;       mphi.RotateZ(phRa);
  TRotation mrot=mphi*mtheta;
  TVector3 dirCkovLORS;
  dirCkovLORS=mrot*dirCkov;
  phiCer  = dirCkovLORS.Phi();                                          //actual value of the phi of the photon
  thetaCer= dirCkovLORS.Theta();                                        //actual value of thetaCerenkov of the photon
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Bool_t AliHMPIDPIDResponse::IsInDead(Float_t x,Float_t y)  {
  
// Check is the current point is outside of sensitive area or in dead zones
// Arguments: x,y -position
// Returns: 1 if not in sensitive zone
             
  for(Int_t iPc=0;iPc<6;iPc++)
    if(x>=fgkMinPcX[iPc] && x<=fgkMaxPcX[iPc] && y>=fgkMinPcY[iPc] && y<=fgkMaxPcY [iPc]) return kFALSE; //in current pc
  
  return kTRUE;
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::Sigma2(Double_t trkTheta,Double_t trkPhi,Double_t ckovTh, Double_t ckovPh) const {
  
// Analithical calculation of total error (as a sum of localization, geometrical and chromatic errors) on Cerenkov angle for a given Cerenkov photon 
// created by a given MIP. Formules according to CERN-EP-2000-058 
// Arguments: Cerenkov and azimuthal angles for Cerenkov photon, [radians]
//            dip and azimuthal angles for MIP taken at the entrance to radiator, [radians]        
//            MIP beta
// Returns: absolute error on Cerenkov angle, [radians]    
  
  TVector3 v(-999,-999,-999);
  Double_t trkBeta = 1./(TMath::Cos(ckovTh)*GetRefIdx());
  
  if(trkBeta > 1) trkBeta = 1;                 //protection against bad measured thetaCer  
  if(trkBeta < 0) trkBeta = 0.0001;            //

  v.SetX(SigLoc (trkTheta,trkPhi,ckovTh,ckovPh,trkBeta));
  v.SetY(SigGeom(trkTheta,trkPhi,ckovTh,ckovPh,trkBeta));
  v.SetZ(SigCrom(trkTheta,ckovTh,ckovPh,trkBeta));

  return v.Mag2();
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::SigLoc(Double_t trkTheta,Double_t trkPhi,Double_t thetaC, Double_t phiC,Double_t betaM) const {
  
// Analitical calculation of localization error (due to finite segmentation of PC) on Cerenkov angle for a given Cerenkov photon 
// created by a given MIP. Fromulae according to CERN-EP-2000-058 
// Arguments: Cerenkov and azimuthal angles for Cerenkov photon, [radians]
//            dip and azimuthal angles for MIP taken at the entrance to radiator, [radians]        
//            MIP beta
// Returns: absolute error on Cerenkov angle, [radians]    
  
  Double_t phiDelta = phiC;

  Double_t sint     = TMath::Sin(trkTheta);
  Double_t cost     = TMath::Cos(trkTheta);
  Double_t sinf     = TMath::Sin(trkPhi);
  Double_t cosf     = TMath::Cos(trkPhi);
  Double_t sinfd    = TMath::Sin(phiDelta);
  Double_t cosfd    = TMath::Cos(phiDelta);
  Double_t tantheta = TMath::Tan(thetaC);
  
  Double_t alpha =cost-tantheta*cosfd*sint;                                                    // formula (11)
  Double_t k = 1.-GetRefIdx()*GetRefIdx()+alpha*alpha/(betaM*betaM);                           // formula (after 8 in the text)
  if (k<0) return 1e10;
  Double_t mu =sint*sinf+tantheta*(cost*cosfd*sinf+sinfd*cosf);                                // formula (10)
  Double_t e  =sint*cosf+tantheta*(cost*cosfd*cosf-sinfd*sinf);                                // formula (9)

  Double_t kk = betaM*TMath::Sqrt(k)/(GapThick()*alpha);                                       // formula (6) and (7)
  Double_t dtdxc = kk*(k*(cosfd*cosf-cost*sinfd*sinf)-(alpha*mu/(betaM*betaM))*sint*sinfd);    // formula (6)           
  Double_t dtdyc = kk*(k*(cosfd*sinf+cost*sinfd*cosf)+(alpha* e/(betaM*betaM))*sint*sinfd);    // formula (7)            pag.4
  
  Double_t errX = 0.2,errY=0.25;                                                                //end of page 7
  return  TMath::Sqrt(errX*errX*dtdxc*dtdxc + errY*errY*dtdyc*dtdyc); 
}
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::SigCrom(Double_t trkTheta,Double_t thetaC, Double_t phiC,Double_t betaM) const {

// Analitical calculation of chromatic error (due to lack of knowledge of Cerenkov photon energy) on Cerenkov angle for a given Cerenkov photon 
// created by a given MIP. Fromulae according to CERN-EP-2000-058 
// Arguments: Cerenkov and azimuthal angles for Cerenkov photon, [radians]
//            dip and azimuthal angles for MIP taken at the entrance to radiator, [radians]        
//            MIP beta
//   Returns: absolute error on Cerenkov angle, [radians]    
  
  Double_t phiDelta = phiC;

  Double_t sint     = TMath::Sin(trkTheta);
  Double_t cost     = TMath::Cos(trkTheta);
  Double_t cosfd    = TMath::Cos(phiDelta);
  Double_t tantheta = TMath::Tan(thetaC);
  
  Double_t alpha =cost-tantheta*cosfd*sint;                                         // formula (11)
  Double_t dtdn = cost*GetRefIdx()*betaM*betaM/(alpha*tantheta);                    // formula (12)
            
  Double_t f = 0.0172*(7.75-5.635)/TMath::Sqrt(24.);

  return f*dtdn;
}//SigCrom()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::SigGeom(Double_t trkTheta,Double_t trkPhi,Double_t thetaC, Double_t phiC,Double_t betaM) const {
  
// Analitical calculation of geometric error (due to lack of knowledge of creation point in radiator) on Cerenkov angle for a given Cerenkov photon 
// created by a given MIP. Formulae according to CERN-EP-2000-058 
// Arguments: Cerenkov and azimuthal angles for Cerenkov photon, [radians]
//            dip and azimuthal angles for MIP taken at the entrance to radiator, [radians]        
//            MIP beta
//   Returns: absolute error on Cerenkov angle, [radians]    

  Double_t phiDelta = phiC;

  Double_t sint     = TMath::Sin(trkTheta);
  Double_t cost     = TMath::Cos(trkTheta);
  Double_t sinf     = TMath::Sin(trkPhi);
  Double_t cosfd    = TMath::Cos(phiDelta);
  Double_t costheta = TMath::Cos(thetaC);
  Double_t tantheta = TMath::Tan(thetaC);
  
  Double_t alpha =cost-tantheta*cosfd*sint;                                                // formula (11)
  
  Double_t k = 1.-GetRefIdx()*GetRefIdx()+alpha*alpha/(betaM*betaM);                       // formula (after 8 in the text)
  if (k<0) return 1e10;

  Double_t eTr = 0.5*RadThick()*betaM*TMath::Sqrt(k)/(GapThick()*alpha);                    // formula (14)
  Double_t lambda = (1.-sint*sinf)*(1.+sint*sinf);                                          // formula (15)

  Double_t c1 = 1./(1.+ eTr*k/(alpha*alpha*costheta*costheta));                             // formula (13.a)
  Double_t c2 = betaM*TMath::Power(k,1.5)*tantheta*lambda/(GapThick()*alpha*alpha);         // formula (13.b)
  Double_t c3 = (1.+eTr*k*betaM*betaM)/((1+eTr)*alpha*alpha);                               // formula (13.c)
  Double_t c4 = TMath::Sqrt(k)*tantheta*(1-lambda)/(GapThick()*betaM);                      // formula (13.d)
  Double_t dtdT = c1 * (c2+c3*c4);
  Double_t trErr = RadThick()/(TMath::Sqrt(12.)*cost);

  return trErr*dtdT;
}//SigGeom()
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::GetNMean(const AliVTrack *vTrk) const {

  // 
  // mean refractive index calculation
  //
  Double_t nmean = -999.; 
  
  Float_t xPc=0.,yPc=0.,thRa=0.,phRa=0.;
  vTrk->GetHMPIDtrk(xPc,yPc,thRa,phRa);
  
  const Int_t ch = vTrk->GetHMPIDcluIdx()/1000000;
  
  const Double_t yRa = yPc - (RadThick()+WinThick()+GapThick())*TMath::Sin(phRa)*TMath::Tan(thRa);  //just linear extrapolation back to RAD
      
  TF1 *RefIndex=0x0;
  
  if(GetRefIndexArray()) RefIndex = (TF1*)(GetRefIndexArray()->At(ch));
  else return nmean;
  
  if(RefIndex) nmean = RefIndex->Eval(yRa);
  else return nmean;
  
  return nmean;   
}  
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
Double_t AliHMPIDPIDResponse::SigmaCorrFact  (Int_t iPart, Double_t occupancy) {

// calculation of sigma correction factor
  
  Double_t corr = 1.0;
       
  switch(iPart) {
    case 0: corr = 0.115*occupancy + 1.166; break; 
    case 1: corr = 0.115*occupancy + 1.166; break;
    case 2: corr = 0.115*occupancy + 1.166; break;
    case 3: corr = 0.065*occupancy + 1.137; break;
    case 4: corr = 0.048*occupancy + 1.202; break;
  }                                                                                                                           
 return corr; 
}

