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


//-------------------------------------------------------------------------
//                      Implementation of   Class AliESDZDC
//   This is a class that summarizes the ZDC data
//   for the ESD   
//   Origin: Christian Klein-Boesing, CERN, Christian.Klein-Boesing@cern.ch 
//-------------------------------------------------------------------------

#include <TMath.h>

#include "AliESDZDC.h"

ClassImp(AliESDZDC)

//______________________________________________________________________________
AliESDZDC::AliESDZDC() :
  AliVZDC(),
  fZDCN1Energy(0),
  fZDCP1Energy(0),
  fZDCN2Energy(0),
  fZDCP2Energy(0),
  fZDCEMEnergy(0),
  fZDCEMEnergy1(0),
  fZDCParticipants(0),
  fZDCPartSideA(0),
  fZDCPartSideC(0),
  fImpactParameter(0),
  fImpactParamSideA(0),
  fImpactParamSideC(0),
  fESDQuality(0),
  fZNCTDChit(kFALSE),
  fZNATDChit(kFALSE),
  fZPCTDChit(kFALSE),
  fZPATDChit(kFALSE),
  fZEM1TDChit(kFALSE),
  fZEM2TDChit(kFALSE)

{
  for(Int_t i=0; i<5; i++){
    fZN1TowerEnergy[i] = fZN2TowerEnergy[i] = 0.;
    fZP1TowerEnergy[i] = fZP2TowerEnergy[i] = 0.;
    fZN1TowerEnergyLR[i] = fZN2TowerEnergyLR[i] = 0.;
    fZP1TowerEnergyLR[i] = fZP2TowerEnergyLR[i] = 0.;
  }
  for(Int_t i=0; i<2; i++){
    fZNACentrCoord[i] = fZNCCentrCoord[i] = 0.;
  }
  for(Int_t i=0; i<32; i++){
    fVMEScaler[i]=0;
    for(Int_t y=0; y<4; y++){
      fZDCTDCData[i][y]=0;
      fZDCTDCCorrected[i][y]=0.;
    }
  }
  for(int it=0; it<7; it++) fZDCTDCChannels[it]=-1;
}

//______________________________________________________________________________
AliESDZDC::AliESDZDC(const AliESDZDC& zdc) :
  AliVZDC(zdc),
  fZDCN1Energy(zdc.fZDCN1Energy),
  fZDCP1Energy(zdc.fZDCP1Energy),
  fZDCN2Energy(zdc.fZDCN2Energy),
  fZDCP2Energy(zdc.fZDCP2Energy),
  fZDCEMEnergy(zdc.fZDCEMEnergy),
  fZDCEMEnergy1(zdc.fZDCEMEnergy1),
  fZDCParticipants(zdc.fZDCParticipants),
  fZDCPartSideA(zdc.fZDCPartSideA),
  fZDCPartSideC(zdc.fZDCPartSideC),
  fImpactParameter(zdc.fImpactParameter),
  fImpactParamSideA(zdc.fImpactParamSideA),
  fImpactParamSideC(zdc.fImpactParamSideC),
  fESDQuality(zdc.fESDQuality),
  fZNCTDChit(zdc.fZNCTDChit),
  fZNATDChit(zdc.fZNATDChit),
  fZPCTDChit(zdc.fZPCTDChit),
  fZPATDChit(zdc.fZPATDChit),
  fZEM1TDChit(zdc.fZEM1TDChit),
  fZEM2TDChit(zdc.fZEM2TDChit)

{
  // copy constructor
  for(Int_t i=0; i<5; i++){
     fZN1TowerEnergy[i] = zdc.fZN1TowerEnergy[i];
     fZN2TowerEnergy[i] = zdc.fZN2TowerEnergy[i];
     fZP1TowerEnergy[i] = zdc.fZP1TowerEnergy[i];
     fZP2TowerEnergy[i] = zdc.fZP2TowerEnergy[i];
     fZN1TowerEnergyLR[i] = zdc.fZN1TowerEnergyLR[i];
     fZN2TowerEnergyLR[i] = zdc.fZN2TowerEnergyLR[i];
     fZP1TowerEnergyLR[i] = zdc.fZP1TowerEnergyLR[i];
     fZP2TowerEnergyLR[i] = zdc.fZP2TowerEnergyLR[i];
  }
  for(Int_t i=0; i<2; i++){
    fZNACentrCoord[i] = zdc.fZNACentrCoord[i];
    fZNCCentrCoord[i] = zdc.fZNCCentrCoord[i];
  }
  for(Int_t i=0; i<32; i++){
    fVMEScaler[i] = zdc.fVMEScaler[i];
    for(Int_t y=0; y<4; y++){
       fZDCTDCData[i][y] = zdc.fZDCTDCData[i][y];
       fZDCTDCCorrected[i][y] = zdc.fZDCTDCCorrected[i][y];
    }
  }
  for(int it=0; it<7; it++) fZDCTDCChannels[it]=zdc.fZDCTDCChannels[it];
}

//______________________________________________________________________________
AliESDZDC& AliESDZDC::operator=(const AliESDZDC&zdc)
{
  // assigment operator
  if(this!=&zdc) {
    AliVZDC::operator=(zdc);
    fZDCN1Energy = zdc.fZDCN1Energy;
    fZDCP1Energy = zdc.fZDCP1Energy;
    fZDCN2Energy = zdc.fZDCN2Energy;
    fZDCP2Energy = zdc.fZDCP2Energy;
    fZDCEMEnergy = zdc.fZDCEMEnergy;
    fZDCEMEnergy1 = zdc.fZDCEMEnergy1;
    for(Int_t i=0; i<5; i++){
       fZN1TowerEnergy[i] = zdc.fZN1TowerEnergy[i];
       fZN2TowerEnergy[i] = zdc.fZN2TowerEnergy[i];
       fZP1TowerEnergy[i] = zdc.fZP1TowerEnergy[i];
       fZP2TowerEnergy[i] = zdc.fZP2TowerEnergy[i];
       fZN1TowerEnergyLR[i] = zdc.fZN1TowerEnergyLR[i];
       fZN2TowerEnergyLR[i] = zdc.fZN2TowerEnergyLR[i];
       fZP1TowerEnergyLR[i] = zdc.fZP1TowerEnergyLR[i];
       fZP2TowerEnergyLR[i] = zdc.fZP2TowerEnergyLR[i];
    }
    //
    fZDCParticipants = zdc.fZDCParticipants;
    fZDCPartSideA = zdc.fZDCPartSideA;
    fZDCPartSideC = zdc.fZDCPartSideC;
    fImpactParameter = zdc.fImpactParameter;
    fImpactParamSideA = zdc.fImpactParamSideA;
    fImpactParamSideC = zdc.fImpactParamSideC;
    //
    for(Int_t i=0; i<2; i++){
         fZNACentrCoord[i] = zdc.fZNACentrCoord[i];
         fZNCCentrCoord[i] = zdc.fZNCCentrCoord[i];
    }
    //
    fESDQuality = zdc.fESDQuality;
    for(Int_t i=0; i<32; i++){
      fVMEScaler[i] = zdc.fVMEScaler[i];
      for(Int_t y=0; y<4; y++){ 
         fZDCTDCData[i][y] = zdc.fZDCTDCData[i][y];
         fZDCTDCCorrected[i][y] = zdc.fZDCTDCCorrected[i][y];
      }
    }
  } 
  fZNCTDChit = zdc.fZNCTDChit;
  fZNATDChit = zdc.fZNATDChit;
  fZPCTDChit = zdc.fZPCTDChit;
  fZPATDChit = zdc.fZPATDChit;
  fZEM1TDChit = zdc.fZEM1TDChit;
  fZEM2TDChit = zdc.fZEM2TDChit;
  
  for(int it=0; it<7; it++) fZDCTDCChannels[it]=zdc.fZDCTDCChannels[it];
 
  return *this;
}

//______________________________________________________________________________
void AliESDZDC::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDZDC *robj = dynamic_cast<AliESDZDC*>(&obj);
  if(!robj)return; // not an AliESDZDC
  *robj = *this;

}


//______________________________________________________________________________
void AliESDZDC::Reset()
{
  // reset all data members
  fZDCN1Energy=0;
  fZDCP1Energy=0;
  fZDCN2Energy=0;
  fZDCP2Energy=0;
  fZDCEMEnergy=0;
  fZDCEMEnergy1=0;
  for(Int_t i=0; i<5; i++){
    fZN1TowerEnergy[i] = fZN2TowerEnergy[i] = 0.;
    fZP1TowerEnergy[i] = fZP2TowerEnergy[i] = 0.;
    fZN1TowerEnergyLR[i] = fZN2TowerEnergyLR[i] = 0.;
    fZP1TowerEnergyLR[i] = fZP2TowerEnergyLR[i] = 0.;
  }
  fZDCParticipants=0;  
  fZDCPartSideA=0;  
  fZDCPartSideC=0;  
  fImpactParameter=0;
  fImpactParamSideA=0;
  fImpactParamSideC=0;
  for(Int_t i=0; i<2; i++){
       fZNACentrCoord[i] = fZNCCentrCoord[i] = 0.;
  }
  fESDQuality=0;
  for(Int_t i=0; i<32; i++){
     fVMEScaler[i] = 0;
     for(Int_t y=0; y<4; y++){
        fZDCTDCData[i][y] = 0;
        fZDCTDCCorrected[i][y] = 0.;
     }
  }
  fZNCTDChit = kFALSE;
  fZNATDChit = kFALSE;
  fZPCTDChit = kFALSE;
  fZPATDChit = kFALSE;
  fZEM1TDChit = kFALSE;
  fZEM2TDChit = kFALSE;

  for(int it=0; it<7; it++) fZDCTDCChannels[it]=-1;

}

//______________________________________________________________________________
void AliESDZDC::Print(const Option_t *) const
{
  //  Print ESD for the ZDC
  printf(" ### ZDC energies: \n");
  printf("\n \t E_ZNC = %1.2f (%1.2f+%1.2f+%1.2f+%1.2f+%1.2f) GeV \n \t E_ZNA = %1.2f (%1.2f+%1.2f+%1.2f+%1.2f+%1.2f) GeV\n"
  " \t E_ZPC = %1.2f GeV E_ZPA = %1.2f GeV"
  "\n E_ZEM1 = %1.2f GeV,   E_ZEM2 = %1.2f GeV\n \n",
  fZDCN1Energy, fZN1TowerEnergy[0], fZN1TowerEnergy[1], fZN1TowerEnergy[2], fZN1TowerEnergy[3], fZN1TowerEnergy[4], 
  fZDCN2Energy, fZN2TowerEnergy[0], fZN2TowerEnergy[1], fZN2TowerEnergy[2], fZN2TowerEnergy[3], fZN2TowerEnergy[4], 
  fZDCP1Energy,fZDCP2Energy, fZDCEMEnergy, fZDCEMEnergy1);
  //
  /*printf(" ### VMEScaler (!=0): \n");
  for(Int_t i=0; i<32; i++) if(fVMEScaler[i]!=0) printf("\t %d \n",fVMEScaler[i]);
  printf("\n");*/
  //
  if(TestBit(AliESDZDC::kTDCcablingSet)) printf(" ### TDC channels: ZNA %d  ZPA %d  ZEM1 %d  ZEM2 %d  ZNC %d  ZPC%d  L0 %d\n\n",
  fZDCTDCChannels[0],fZDCTDCChannels[1],fZDCTDCChannels[2],fZDCTDCChannels[3],fZDCTDCChannels[4],fZDCTDCChannels[5],fZDCTDCChannels[6]);
  /*for(Int_t i=0; i<32; i++){
    for(Int_t j=0; j<4; j++)
      if(TMath::Abs(fZDCTDCCorrected[i][j])>1e-4) printf("\t %1.0f \n",fZDCTDCCorrected[i][j]);
  }*/
  printf("\n");
}

//______________________________________________________________________________
Bool_t AliESDZDC::GetZNCentroidInPbPb(Float_t beamEne, Double_t centrZNC[2], Double_t centrZNA[2]) 
{
  // Provide coordinates of centroid over ZN (side C) front face
  if(beamEne==0){
    printf(" ZDC centroid in PbPb can't be calculated with E_beam = 0 !!!\n");
    for(Int_t jj=0; jj<2; jj++) fZNCCentrCoord[jj] = 999.;
    return kFALSE;
  }

  const Float_t x[4] = {-1.75, 1.75, -1.75, 1.75};
  const Float_t y[4] = {-1.75, -1.75, 1.75, 1.75};
  const Float_t alpha=0.395;
  Float_t numXZNC=0., numYZNC=0., denZNC=0., cZNC, wZNC; 
  Float_t numXZNA=0., numYZNA=0., denZNA=0., cZNA, wZNA; 
  //
  for(Int_t i=0; i<4; i++){
    if(fZN1TowerEnergy[i+1]>0.) {
      wZNC = TMath::Power(fZN1TowerEnergy[i+1], alpha);
      numXZNC += x[i]*wZNC;
      numYZNC += y[i]*wZNC;
      denZNC += wZNC;
    }
    if(fZN2TowerEnergy[i+1]>0.) {
      wZNA = TMath::Power(fZN2TowerEnergy[i+1], alpha);
      numXZNA += x[i]*wZNA;
      numYZNA += y[i]*wZNA;
      denZNA += wZNA;
    }
  }
  //
  if(denZNC!=0){
    Float_t nSpecnC = fZDCN1Energy/beamEne;
    cZNC = 1.89358-0.71262/(nSpecnC+0.71789);
    fZNCCentrCoord[0] = cZNC*numXZNC/denZNC;
    fZNCCentrCoord[1] = cZNC*numYZNC/denZNC;
  } 
  else{
    fZNCCentrCoord[0] = fZNCCentrCoord[1] = 999.;
  }
  if(denZNA!=0){
    Float_t nSpecnA = fZDCN2Energy/beamEne;
    cZNA = 1.89358-0.71262/(nSpecnA+0.71789);
    fZNACentrCoord[0] = cZNA*numXZNA/denZNA;
    fZNACentrCoord[1] = cZNA*numYZNA/denZNA;
  } 
  else{
    fZNACentrCoord[0] = fZNACentrCoord[1] = 999.;
  }
  //
  for(Int_t il=0; il<2; il++){
    centrZNC[il] = fZNCCentrCoord[il];
    centrZNA[il] = fZNACentrCoord[il];
  }
  
  return kTRUE;
}

//______________________________________________________________________________
Bool_t AliESDZDC::GetZNCentroidInpp(Double_t centrZNC[2], Double_t centrZNA[2]) 
{
  // Provide coordinates of centroid over ZN (side C) front face
  const Float_t x[4] = {-1.75, 1.75, -1.75, 1.75};
  const Float_t y[4] = {-1.75, -1.75, 1.75, 1.75};
  const Float_t alpha=0.5;
  Float_t numXZNC=0., numYZNC=0., denZNC=0., wZNC; 
  Float_t numXZNA=0., numYZNA=0., denZNA=0., wZNA; 
  //
  for(Int_t i=0; i<4; i++){
    if(fZN1TowerEnergy[i+1]>0.) {
      wZNC = TMath::Power(fZN1TowerEnergy[i+1], alpha);
      numXZNC += x[i]*wZNC;
      numYZNC += y[i]*wZNC;
      denZNC += wZNC;
    }
    if(fZN2TowerEnergy[i+1]>0.) {
      wZNA = TMath::Power(fZN2TowerEnergy[i+1], alpha);
      numXZNA += x[i]*wZNA;
      numYZNA += y[i]*wZNA;
      denZNA += wZNA;
    }
  }
  //
  if(denZNC!=0){
    fZNCCentrCoord[0] = numXZNC/denZNC;
    fZNCCentrCoord[1] = numYZNC/denZNC;
  } 
  else{
    fZNCCentrCoord[0] = fZNCCentrCoord[1] = 999.;
  }
  if(denZNA!=0){
    fZNACentrCoord[0] = numXZNA/denZNA;
    fZNACentrCoord[1] = numYZNA/denZNA;
  } 
  else{
    fZNACentrCoord[0] = fZNACentrCoord[1] = 999.;
  }
  //
  for(Int_t il=0; il<2; il++){
    centrZNC[il] = fZNCCentrCoord[il];
    centrZNA[il] = fZNACentrCoord[il];
  }
  
  return kTRUE;
}

//______________________________________________________________________________
Float_t AliESDZDC::GetZNTDCSum(Int_t ihit) const
{
    if(ihit>4 || !(TestBit(AliESDZDC::kCorrectedTDCFilled))){
      return 1000.; // only up to 4 hits are stored && return sum only for calibrated TDCs
    }
    else{
      if(!(TestBit(AliESDZDC::kTDCcablingSet))){ // RUN1: data cabled ch. hardwired in the code
         if((fZDCTDCData[10][ihit]!=0) && (fZDCTDCData[12][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[10][ihit]+fZDCTDCCorrected[12][ihit]);
	 else return 999.;
      }
      else{ // RUN2: everything done from mapping
         if(fZDCTDCChannels[4]<0 ||  fZDCTDCChannels[2]<0){// RUN2 data but without signal code!!! 
           if((fZDCTDCData[16][ihit]!=0) && (fZDCTDCData[18][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[16][ihit]+fZDCTDCCorrected[18][ihit]);
           else return 998.;
	 }
	 else{
	   if((fZDCTDCData[fZDCTDCChannels[4]][ihit]!=0) && (fZDCTDCData[fZDCTDCChannels[2]][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[fZDCTDCChannels[4]][ihit]+fZDCTDCCorrected[fZDCTDCChannels[2]][ihit]);
           else return 997.;
	 }
      }
   }
}

//______________________________________________________________________________
Float_t AliESDZDC::GetZNTDCDiff(Int_t ihit) const
{
    if(ihit>4 || !(TestBit(AliESDZDC::kCorrectedTDCFilled))){
      return 1000.; // only up to 4 hits are stored && return sum only for calibrated TDCs
    }
    else{
      if(!(TestBit(AliESDZDC::kTDCcablingSet))){ // RUN1: data cabled ch. hardwired in the code
         if((fZDCTDCData[10][ihit]!=0) && (fZDCTDCData[12][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[10][ihit]-fZDCTDCCorrected[12][ihit]);
	 else return 999.;
      }
      else{ // RUN2: everything done from mapping
         if(fZDCTDCChannels[4]<0 ||  fZDCTDCChannels[2]<0){// RUN2 data but without signal code!!! 
           if((fZDCTDCData[16][ihit]!=0) && (fZDCTDCData[18][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[16][ihit]-fZDCTDCCorrected[18][ihit]);
           else return 998.;
	 }
	 else{
	   if((fZDCTDCData[fZDCTDCChannels[4]][ihit]!=0) && (fZDCTDCData[fZDCTDCChannels[2]][ihit]!=0)) return (Float_t) (fZDCTDCCorrected[fZDCTDCChannels[2]][ihit]-fZDCTDCCorrected[fZDCTDCChannels[4]][ihit]);
           else return 997.;
	 }
      }
   }
}
