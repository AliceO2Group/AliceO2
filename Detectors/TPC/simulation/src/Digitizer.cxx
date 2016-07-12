/// \file AliTPCUpgradeDigitizer.cxx
/// \brief Digitizer for the TPC
#include "Digitizer.h"
#include "Point.h"                // for Point
#include "TRandom.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray
#include "TCollection.h"          // for TIter
#include "Mapper.h"
#include "TMath.h"
#include <cmath>

#include <iostream>

ClassImp(AliceO2::TPC::Digitizer)

using namespace AliceO2::TPC;

Digitizer::Digitizer():
TObject(),
mDigitContainer(nullptr),
mGain(2000.)
{}

Digitizer::~Digitizer(){
  if(mDigitContainer) delete mDigitContainer;
}

void Digitizer::init(){
  mDigitContainer = new DigitContainer();
}

DigitContainer *Digitizer::Process(TClonesArray *points){
  mDigitContainer->reset();
  
  Int_t nEle;
  Float_t posEle[4] = {0,0,0,0};
  
  const Mapper& mapper = Mapper::instance();
    
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *inputpoint = dynamic_cast<Point *>(*pointiter);
    
    // Convert energy loss in number of electrons
    nEle = (Int_t)(inputpoint->GetEnergyLoss()*1000000/37.3);
    
    posEle[0] = inputpoint->GetX();
    posEle[1] = inputpoint->GetY();
    posEle[2] = inputpoint->GetZ();
    posEle[3] = nEle;
    
    const GlobalPosition3D pos (posEle[0], posEle[1], posEle[2]);
    const DigitPos digiPos = mapper.findDigitPosFromGlobalPosition(pos);
    
    // remove hits that are out of the acceptance of the TPC
    if(!digiPos.isValid()) continue;
    if(abs(posEle[2]) > 250) continue;
    
    // Compute mean time Bin of the charge cloud
    const Int_t timeBin = getTimeBin(posEle[2]);
    
    Float_t nEleAmp = 0;
    
    //Loop over electrons - take into account diffusion and amplification
    for(Int_t iEle=0; iEle < nEle; ++iEle){
      
      //attachment
      Float_t time = (250-abs(posEle[2])/2.58); // in us
      Float_t attProb = 250. * 5.e-6 * time;
      // Float_t attProb = fTPCParam->GetAttCoef()*fTPCParam->GetOxyCont()*time;
      if((gRandom->Rndm(0)) < attProb) continue;
      
      getElectronDrift(posEle);
      
      nEleAmp += getGEMAmplification();
    }
    //end of loop over electrons
    
    // here the time response and sorting of the charge spread due to diffusion into different pad rows, pads and time bins will come...
    
    Int_t ADCvalue = getADCvalue(nEleAmp);

    mDigitContainer->addDigit(digiPos.getCRU().number(), digiPos.getPadPos().getRow(), digiPos.getPadPos().getPad(), timeBin, ADCvalue);
  }
  return mDigitContainer;
}

void Digitizer::getElectronDrift(Float_t *posEle){
  Float_t driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=TMath::Sqrt(driftl);
  Float_t DiffT = 0.0209;
  Float_t DiffL = 0.0221;
  Float_t sigT = driftl*DiffT;
  Float_t sigL = driftl*DiffL;
  posEle[0]=gRandom->Gaus(posEle[0],sigT);
  posEle[1]=gRandom->Gaus(posEle[1],sigT);
  posEle[2]=gRandom->Gaus(posEle[2],sigL);
}

Int_t Digitizer::getADCvalue(Float_t nElectrons){
  // parameters to be stored someplace else
  Float_t ADCSat = 1024;
  Float_t Qel = 1.602e-19;
  Float_t ChipGain = 12;
  Float_t ADCDynRange = 2000;
  
  Float_t ADCvalue = nElectrons*Qel*1.e15*ChipGain*ADCSat/ADCDynRange;    
  ADCvalue = TMath::Nint(ADCvalue);
  if(ADCvalue >= ADCSat) ADCvalue = ADCSat - 1; // saturation
  return ADCvalue;
}

Float_t Digitizer::getGEMAmplification(){
  Float_t rn=TMath::Max(gRandom->Rndm(0),1.93e-22);
  Float_t signal = (Int_t)(-(mGain) * TMath::Log(rn));
  return signal;
}

const Int_t Digitizer::getTimeBin(Float_t zPos){
  //TODO: parameterize the conversion of the zPos to timeBin + should go someplace else.
  Int_t timeBin = (Int_t)((250-abs(zPos))/(2.58*0.2));
  return timeBin;
}

Double_t  Digitizer::Gamma4(Double_t x, Double_t p0, Double_t p1){
  // should of course go someplace else!
  if (x<0) return 0;
  Double_t g1 = TMath::Exp(-4.*x/p1);
  Double_t g2 = TMath::Power(x/p1,4);
  return p0*g1*g2;
}