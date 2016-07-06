/// \file AliTPCUpgradeDigitizer.cxx
/// \brief Digitizer for the TPC
#include "Digitizer.h"
#include "Point.h"                // for Point
#include "TRandom.h"

#include "FairLogger.h"           // for LOG
#include "TClonesArray.h"         // for TClonesArray
#include "TCollection.h"          // for TIter
#include "Mapper.h"
#include <cmath>

#include <iostream>

ClassImp(AliceO2::TPC::Digitizer)

using namespace AliceO2::TPC;

Digitizer::Digitizer():
TObject(),
mDigitContainer(nullptr),
mGain(1.)
{}

Digitizer::~Digitizer(){
  if(mDigitContainer) delete mDigitContainer;
}

void Digitizer::Init(){
  mDigitContainer = new DigitContainer();
}

DigitContainer *Digitizer::Process(TClonesArray *points){
  mDigitContainer->Reset();
 
  // Directly write the Hits to file
  Int_t i=10;
  Int_t nEle;
  Float_t posEle[4] = {0,0,0,0};
  
  const Mapper& mapper = Mapper::instance();
  
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *inputpoint = dynamic_cast<Point *>(*pointiter);
    
    // Convert energyloss in number of electrons
    nEle=0;
    //TODO: set proper parameters somewhere
//     nEle = (Int_t)(((inputpoint->GetEnergyLoss())-poti)/wIon) + 1;
    nEle = (Int_t)(inputpoint->GetEnergyLoss()*1000000/38.1);
    posEle[0] = inputpoint->GetX();
    posEle[1] = inputpoint->GetY();
    posEle[2] = inputpoint->GetZ();
    posEle[3] = nEle;
    
    //Loop over electrons - take into account diffusion and amplification
    for(Int_t iEle=0; iEle < nEle; ++iEle){
      //attachment
//       if((gRandom->Rndm(0)) < tpcparam->GetAttachmentCoeff()) continue;
      DriftElectrons(posEle);
      GEMAmplification(1);

    }
    
    
    //TODO: convert x & y to pad and row coordinate. z coordinate -> drift. charge -> adc value
    const GlobalPosition3D pos (inputpoint->GetX(), inputpoint->GetY(), inputpoint->GetZ());
    const DigitPos digiPos = mapper.findDigitPosFromGlobalPosition(pos);
    if(!digiPos.isValid()) continue;
//     const Int_t timeBin = ((abs(inputpoint->GetZ())-250)/(2.58*0.2));
    //TODO: proper timebin
    const Int_t timeBin = 100;
    
    mDigitContainer->AddDigit(digiPos.getCRU().number(), digiPos.getPadPos().getRow(), digiPos.getPadPos().getPad(), timeBin, inputpoint->GetEnergyLoss());
  }
  return mDigitContainer;
}

void Digitizer::DriftElectrons(Float_t *posEle){
//   Float_t driftl=posEle[2];
//   if(driftl<0.01) driftl=0.01;
//   driftl=TMath::Sqrt(driftl);
//   Float_t sigT = driftl*(fTPCParam->GetDiffT());
//   Float_t sigL = driftl*(fTPCParam->GetDiffL());
//   posEle[0]=gRandom->Gaus(posEle[0],sigT);
//   posEle[1]=gRandom->Gaus(posEle[1],sigT);
//   posEle[2]=gRandom->Gaus(posEle[2],sigL);
}

void Digitizer::GEMAmplification(Float_t nEle){
  nEle *= 10;
}