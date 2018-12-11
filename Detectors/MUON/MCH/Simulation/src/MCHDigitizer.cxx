// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHSimulation/MCHDigitizer.h"

#include "TMath.h"
#include "TProfile2D.h"
#include "TRandom.h"
#include <algorithm>
#include <cassert>

using namespace o2::mch;

ClassImp(o2::mch::MCHDigitizer);

void MCHDigitizer::init()
{
// initialize the array of detector segmentation's
  for(Int_t i=0; i<mNdE; ++i){
    mSegbend[i]= Segmentation(i,kTRUE);
    mSegnon[i] = Segmentation(i,kFALSE);
    
  }
  
  
  // To be done:
  //0) adding processing steps and proper translation of charge to adc counts
  //need for "sdigits" (one sdigit per Hit in aliroot) vs. digits (comb. signal per pad) two steps
  //1) differentiate between chamber types for signal generation:
  //2) add initialisation of parameters to be set for digitisation (pad response, hard-ware) at central place
  //3) add a test
  //4) check read-out chain efficiency handling in current code: simple efficiency values? masking?
  //different strategy?
  //5) handling of time dimension: what changes w.r.t. aliroot check HMPID
  //6) handling of MCtruth information
 
  //TODO time dimension
  //can one avoid these initialisation with this big for-loop as TOF?
}

//______________________________________________________________________

void MCHDigitizer::process(const std::vector<Hit>* hits, std::vector<Digit>* digits)
{
  // hits array of MCH hits for a given simulated event
  for (auto& hit : *hits) {
    //TODO: check if change for time structure
    processHit(hit, mEventTime);
   } // end loop over hits
  //TODO: merge (new member function, add charge) of digits that are on same pad:
  //things to think about in terms of time costly

    digits->clear();
    fillOutputContainer(*digits);

}

//______________________________________________________________________

Int_t MCHDigitizer::processHit(const Hit &hit,Double_t event_time)
{

  //hit position(cm)
  Float_t pos[3] = { hit.GetX(), hit.GetY(), hit.GetZ() };
  //convert energy to charge, float enough?
  Float_t charge = etocharge(hit.GetEnergyLoss());
  //time information
  Float_t time = hit.GetTime();//how to trace
  Int_t detID = hit.GetDetectorID();
  //# digits for hit
  Int_t ndigits=0;
  
  Float_t anodpos = getAnod(pos[0],detID);

  //TODO: charge sharing between planes,
  //possibility to do random seeding in controlled way
  // be able to be 100% reproducible if wanted? or already given up on geant level?
  //signal will be around neighbouring anode-wire 
  //distance of impact point and anode, needed for charge sharing
  Float_t anoddis = TMath::Abs(pos[0]-anodpos);
  //question on how to retrieve charge fraction deposited in both half spaces
  //throw a dice?
  //should be related to electrons fluctuating out/in one/both halves (independent x)
  //  Float_t fracplane = 0.5;//to be replaced by function of annodis
  Float_t fracplane = chargeCorr();//should become a function of anoddis
  Float_t chargebend= fracplane*charge;
  Float_t chargenon = charge/fracplane;
  //last line  from Aliroot, not understood why
  //since charge = charchbend+chargenon and not multiplication
  Float_t signal = 0.0;

  //borders of charge gen. 
  Double_t xMin = anodpos-mQspreadX*0.5;
  Double_t xMax = anodpos+mQspreadX*0.5;

  Double_t yMin = pos[1]-mQspreadY*0.5;
  Double_t yMax = pos[1]+mQspreadY*0.5;
  
  //pad-borders
  Float_t xmin =0.0;
  Float_t xmax =0.0;
  Float_t ymin =0.0;
  Float_t ymax =0.0;
 
  //use DetectorID to get area for signal induction               
  // SegmentationImpl3.h: Return the list of paduids for the pads contained in the box {xmin,ymin,xmax,ymax}.      
  //  std::vector<int> getPadUids(double xmin, double ymin, double xmax, double ymax) const;
  //is this available via Segmentation.h interface already?

  //testing with only one pad
  //  Int_t paduidbend = mSegbend[detID].findPadByPosition(pos[0],pos[1]);
  //  Int_t paduidnon  = mSegnon[detID].findPadByPosition(pos[0],pos[1]);
  //correct coordinate system? how misalignment enters?


  /*mPadIDsbend = mSegbend.getPadUids(xMin,xMax,yMin,yMax);
  mPadIDsnon  = mSegnon.getPadUids(xMin,xMax,yMin,yMax);
  */
  for(auto & padidbend : mPadIDsbend){
    //retrieve coordinates for each pad
    xmin =  mSegbend.padPositionX(padidbend)-mSegBend.padSizeX(padidbend)*0.5;
    xmax =  mSegbend.padPositionX(padidbend)+mSegbend.padSizeX(padidbend)*0.5;
    ymin =  mSegbend.padPositionY(padidbend)-mSegBend.padSizeY(padidbend)*0.5;
    ymax =  mSegbend.padPositionY(padidbend)+mSegbend.padSizeY(padidbend)*0.5;
        
    // 1st step integrate induced charge for each pad
    signal = chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargebend);
    if(signal>mChargeThreshold && signal<mChargeSat){
      //2nd condition in Aliroot said to be only for backward compatibility
      //to be seen...means that there is no digit, if signal above... strange!
      //2n step TODO: pad response function, electronic response
      signal = response(detID,signal);
      mDigits.emplace_back(padidbend,signal);//how trace time?
      ++ndigits;
    }
  }

  for(auto & padidnon : mPadIDsnon){
    //retrieve coordinates for each pad
    xmin =  mSegnon.padPositionX(padidnon)-mSegnon.padSizeX(padidnon)*0.5;
    xmax =  mSegnon.padPositionX(padidnon)+mSegnon.padSizeX(padidnon)*0.5;
    ymin =  mSegnon.padPositionY(padidnon)-mSegnon.padSizeY(padidnon)*0.5;
    ymax =  mSegnon.padPositionY(padidnon)+mSegnon.padSizeY(padidnon)*0.5;
       
    //retrieve charge for given x,y with Mathieson
    signal = chargePad(anodpos,pos[1],xmin,xmax,ymin,ymax,detID,chargenon);
    if(signal>mChargeThreshold && signal<mChargeSat){
      signal = response(detID,signal);
      mDigits.emplace_back(padidnon,signal);//how is time propagated
      ++ndigits;
    }
  }	
    
  //OLD only single pad mDigits.emplace_back(paduidbend,chargebend, time);
  //OLD only single pad mDigits.emplace_back(paduidnon, chargenon, time);
  return ndigits;
}
//_____________________________________________________________________
Float_t MCHDigitizer::etocharge(Float_t edepos){
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
Double_t MCHDigitizer::chargePad(Float_t x, Float_t y, Float_t xmin, Float_t xmax, Float_t ymin, Float_t ymax, Int_t detID, Float_t charge ){
  //see AliMUONResponseV0.cxx (inside DisIntegrate)
  // and AliMUONMathieson.cxx (IntXY)
  Int_t station = 0;
  if(detID>11) station = 1;
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
Double_t MCHDigitizer::response(Float_t charge, Int_t detID){
  //to be done: calculate from energed
  return charge;
}
//______________________________________________________________________
Float_t MCHDigitizer::getAnod(Float_t x, Int_t detID){

  Float_t pitch = mInversePitch[1];
  if(detID<11) pitch = mInversePitch[0];
  
  Int_t n = Int_t(x/pitch);
  Float_t wire = (x>0) ? n+0.5 : n-0.5;
  return pitch*wire;
}
//______________________________________________________________________
Float_t MCHDigitizer::chargeCorr(){
  //taken from AliMUONResponseV0
  //conceptually not at all understood why this should make sense
  //mChargeCorr not taken
  return TMath::Exp(gRandom->Gaus(0.0, mChargeCorr/2.0));
}
//______________________________________________________________________
//not clear if needed for DPL or modifications required
void MCHDigitizer::fillOutputContainer(std::vector<Digit>& digits)
{
  // filling the digit container
  if (mDigits.empty())
    return;
  
  auto itBeg = mDigits.begin();
  auto iter = itBeg;
  for (; iter != mDigits.end(); ++iter) {
    digits.emplace_back(*iter);
  }
  
  mDigits.erase(itBeg, iter);
}
//______________________________________________________________________
void MCHDigitizer::flushOutputContainer(std::vector<Digit>& digits)
{ // flush all residual buffered data
  //not clear if neede in DPL
  fillOutputContainer(digits);
}

