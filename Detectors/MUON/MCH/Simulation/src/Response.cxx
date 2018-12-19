// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/Response.h"

#include "TMath.h"
#include "TRandom.h"

using namespace o2::mch;

//_____________________________________________________________________
Float_t Response::etocharge(Float_t edepos){
  //Todo convert in charge in number of electrons
  //equivalent if IntPH in AliMUONResponseV0 in Aliroot
  //to be clarified:
  //1) why effective parameterisation with Log?
  //2) any will to provide random numbers
  //3) Float in aliroot, Double needed?
  //with central seed to be reproducible?
  //TODO: dependence on station
  //TODO: check slope meaning in thesis
  Int_t nel = Int_t(edepos*1.e9/27.4);
  Float_t charge=0;
  if (nel ==0) nel=1;
  for (Int_t i=1; i<=nel;i++) {
    Float_t arg=0.;
    while(!arg) arg = gRandom->Rndm();
    charge -= mChargeSlope*TMath::Log(arg);
    
  }
  return charge;
}
//_____________________________________________________________________
Double_t Response::chargePad(Float_t x, Float_t y, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Int_t detID, Float_t charge ){
  //see AliMUONResponseV0.cxx (inside DisIntegrate)
  // and AliMUONMathieson.cxx (IntXY)
  Int_t station = 0;
  if(detID>299) station = 1;//wrong numbers!
  //correct? should take info from segmentation
  // normalise w.r.t. Pitch
  xmin *= mInversePitch[station];
  xmax *= mInversePitch[station];
  ymin *= mInversePitch[station];
  ymax *= mInversePitch[station];
  // The Mathieson function
  Double_t ux1=mSqrtK3x[station]*TMath::TanH(mK2x[station]*xmin);
  Double_t ux2=mSqrtK3x[station]*TMath::TanH(mK2x[station]*xmax);
  
  Double_t uy1=mSqrtK3y[station]*TMath::TanH(mK2y[station]*ymin);
  Double_t uy2=mSqrtK3y[station]*TMath::TanH(mK2y[station]*ymax);
  
  return 4.*mK4x[station]*(TMath::ATan(ux2)-TMath::ATan(ux1))*
    mK4y[station]*(TMath::ATan(uy2)-TMath::ATan(uy1))*charge;
}
//______________________________________________________________________
Double_t Response::response(Float_t charge, Int_t detID){
  //to be done: calculate from induced charge signal
  return charge;
}
//______________________________________________________________________
Float_t Response::getAnod(Float_t x, Int_t detID){

  Float_t pitch = mInversePitch[1];
  if(detID<299) pitch = mInversePitch[0]; //guess for numbers!
  
  Int_t n = Int_t(x/pitch);
  Float_t wire = (x>0) ? n+0.5 : n-0.5;
  return pitch*wire;
}
//______________________________________________________________________
Float_t Response::chargeCorr(){
  //taken from AliMUONResponseV0
  //conceptually not at all understood why this should make sense
  //mChargeCorr not taken
  return TMath::Exp(gRandom->Gaus(0.0, mChargeCorr/2.0));
}


