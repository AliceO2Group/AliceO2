/// \file Digitizer.cxx
/// \brief Digitizer for the TPC
#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/GEMAmplification.h"

#include "TPCBase/Mapper.h"

#include "TRandom.h"
#include "TF1.h"
#include "TClonesArray.h"
#include "TCollection.h"
#include "TMath.h"

#include "FairLogger.h"

#include <iostream>

ClassImp(AliceO2::TPC::Digitizer)

using namespace AliceO2::TPC;

Digitizer::Digitizer()
  : TObject(),
    mDigitContainer(nullptr),
    mRandomGaus()
{}

Digitizer::~Digitizer()
{
  delete mDigitContainer;
}

void Digitizer::init()
{
  /// @todo get rid of new? check with Mohammad
  mDigitContainer = new DigitContainer();
  
  mRandomGaus.initialize(RandomRing::RandomType::Gaus);
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
  /// @todo Containers?
  mDigitContainer->reset();
  
  RandomRing randomFlat(RandomRing::RandomType::Flat);
  /// @todo GlobalPosition3d + nEle instead?
  Float_t posEle[4] = {0., 0., 0., 0.};

  const Mapper& mapper = Mapper::instance();

  GEMAmplification g(EFFGAINGEM1, EFFGAINGEM2, EFFGAINGEM3, EFFGAINGEM4);
  
  for (TIter pointiter = TIter(points).Begin(); pointiter != TIter::End(); ++pointiter) {
    Point *inputpoint = static_cast<Point *>(*pointiter);
    
    posEle[0] = inputpoint->GetX();
    posEle[1] = inputpoint->GetY();
    posEle[2] = inputpoint->GetZ();
    posEle[3] = static_cast<int>(inputpoint->GetEnergyLoss()/WION);
    
    //Loop over electrons
    /// @todo can be vectorized?
    /// @todo split transport and signal formation in two separate loop?
    for(Int_t iEle=0; iEle < posEle[3]; ++iEle) {
      
      // Attachment
      /// @todo simple scaling possible outside this loop? or after diffusion...
      const Float_t attProb = ATTCOEF * OXYCONT * getTime(posEle[2]);
      if(randomFlat.getNextValue() < attProb) continue;

      // Drift and Diffusion
      ElectronDrift(posEle);

      // remove electrons that end up outside the active volume
      /// @todo should go to mapper?
      if(TMath::Abs(posEle[2]) > TPCLENGTH) continue;

      const GlobalPosition3D posElePad (posEle[0], posEle[1], posEle[2]);
      const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posElePad);
       
      if(!digiPadPos.isValid()) continue;

      const Int_t nElectronsGEM = g.getStackAmplification();
      
      // Loop over all individual pads with signal due to pad response function
      /// @todo modularization -> own class
      std::vector<PadResponse> padResponseContainer;
      getPadResponse(posEle[0], posEle[1], padResponseContainer);
      for(auto &padresp : padResponseContainer ) {
        const Int_t pad = digiPadPos.getPadPos().getPad() + padresp.getPad();
        const Int_t row = digiPadPos.getPadPos().getRow() + padresp.getRow();
        const Float_t weight = padresp.getWeight();
        
        /// @todo Time management in continuous mode (adding the time of the event?)
        const Float_t startTime = getTime(posEle[2]);
//         std::cout << startTime << " " << getTimeBinFromTime(startTime) << " " << getTimeFromBin(int(getTimeBinFromTime(startTime) + 0.5)) << " " << getTimeFromBin(getTimeBinFromTime(startTime) + 0.5) << " " << posEle[2] << "\n";
        
        // Loop over all time bins with signal due to time response
        for(Float_t bin = 0; bin<5; ++bin) {
          /// @todo check how the SAMPA digitisation is applied
          /// @todo modularization -> own SAMPA class
          /// @todo conversion to ADC counts already here? How about saturation?
          Double_t signal = 55*Gamma4(startTime+bin*ZBINWIDTH, startTime, nElectronsGEM*weight);
//           Double_t signal = 55*Gamma4(getTimeFromBin(getTimeBinFromTime(startTime) + bin + 0.5), startTime, nElectronsGEM*weight);
          if(signal <= 0.) continue;
          mDigitContainer->addDigit(digiPadPos.getCRU().number(), getTimeBinFromTime(startTime+bin*ZBINWIDTH), row, pad, signal);
//           std::cout << getTimeBinFromTime(getTimeFromBin(getTimeBinFromTime(startTime) + bin + 0.5)) << "\n";
        }
        // end of loop over time bins
      }
      // end of loop over pads
    }
    //end of loop over electrons
  }
  // end of loop over points

  return mDigitContainer;
}


void Digitizer::ElectronDrift(Float_t *posEle)
{
  Float_t driftl=posEle[2];
  if(driftl<0.01) driftl=0.01;
  driftl=TMath::Sqrt(driftl);
  Float_t sigT = driftl*DIFFT;
  Float_t sigL = driftl*DIFFL;
  posEle[0]=(mRandomGaus.getNextValue() * sigT) + posEle[0];
  posEle[1]=(mRandomGaus.getNextValue() * sigT) + posEle[1];
  posEle[2]=(mRandomGaus.getNextValue() * sigL) + posEle[2];
}