/// \file Digitizer.cxx
/// \author Andi Mathis, andreas.mathis@ph.tum.de

#include "TPCSimulation/Digitizer.h"
#include "TPCSimulation/Point.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"
#include "TPCSimulation/GEMAmplification.h"
#include "TPCSimulation/ElectronTransport.h"
#include "TPCSimulation/SAMPAProcessing.h"

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
  : TObject()
  , mDigitContainer(nullptr)
{}

Digitizer::~Digitizer()
{
  delete mDigitContainer;
}

void Digitizer::init()
{
  /// @todo get rid of new? check with Mohammad
  mDigitContainer = new DigitContainer();
}

DigitContainer *Digitizer::Process(TClonesArray *points)
{
  /// @todo Containers?
  mDigitContainer->reset();
  
  RandomRing randomFlat(RandomRing::RandomType::Flat);

  const Mapper& mapper = Mapper::instance();

  // static_thread for thread savety?
  // avid multiple creation of the random lookup tables inside
  static GEMAmplification gemStack(EFFGAINGEM1, EFFGAINGEM2, EFFGAINGEM3, EFFGAINGEM4);
  static ElectronTransport electronTransport;

  static std::vector<PadResponse> padResponseContainer;
  
  for (auto pointObject: *points) {
    Point *inputpoint = static_cast<Point *>(pointObject);
    
    const GlobalPosition3D posEle(inputpoint->GetX(), inputpoint->GetY(), inputpoint->GetZ());
    int nPrimaryElectrons = static_cast<int>(inputpoint->GetEnergyLoss()/WION);
    
    int MCEventID = inputpoint->GetEventID();
    int MCTrackID = inputpoint->GetTrackID();

    //Loop over electrons
    /// @todo can be vectorized?
    /// @todo split transport and signal formation in two separate loop?
    for(int iEle=0; iEle < nPrimaryElectrons; ++iEle) {
            
      // Drift and Diffusion
      const GlobalPosition3D posEleDiff = electronTransport.getElectronDrift(posEle);
      
      /// @todo Time management in continuous mode (adding the time of the event?)
      const float driftTime = getTime(posEleDiff.getZ());

      // Attachment
      if(randomFlat.getNextValue() < electronTransport.getAttachmentProbability(driftTime)) continue;

      // remove electrons that end up outside the active volume
      /// @todo should go to mapper?
      if(fabs(posEleDiff.getZ()) > TPCLENGTH) continue;

      const DigitPos digiPadPos = mapper.findDigitPosFromGlobalPosition(posEleDiff);
      if(!digiPadPos.isValid()) continue;

      const int nElectronsGEM = gemStack.getStackAmplification();
      
      // Loop over all individual pads with signal due to pad response function
      /// @todo modularization -> own class

      getPadResponse(posEleDiff.getX(), posEleDiff.getY(), padResponseContainer);
      for(auto &padresp : padResponseContainer ) {
        const int pad = digiPadPos.getPadPos().getPad() + padresp.getPad();
        const int row = digiPadPos.getPadPos().getRow() + padresp.getRow();
        const float weight = padresp.getWeight();
        
        // Loop over all time bins with signal due to time response
//         for(float bin = 0; bin<5; ++bin) {
//           /// @todo check how the SAMPA digitisation is applied
//           /// @todo modularization -> own SAMPA class
//           /// @todo conversion to ADC counts already here? How about saturation?
//           Double_t signal = SAMPAProcessing::getGamma4(driftTime+bin*ZBINWIDTH, driftTime, nElectronsGEM*weight);
// //           Double_t signal = 55*Gamma4(getTimeFromBin(getTimeBinFromTime(driftTime) + bin + 0.5), driftTime, nElectronsGEM*weight);
//           if(signal <= 0.) continue;
//           mDigitContainer->addDigit(MCEventID, MCTrackID, digiPadPos.getCRU().number(), getTimeBinFromTime(driftTime+bin*ZBINWIDTH), row, pad, signal);
//         }

        // 8 chosen to fit with SSE registers
        for(float bin = 0; bin<8; bin+=Vc::float_v::Size) {
          /// @todo check how the SAMPA digitisation is applied
          Vc::float_v binvector;
          for (int i=0;i<Vc::float_v::Size;++i) { binvector[i]=bin+i; }
          Vc::float_v time = driftTime + binvector*ZBINWIDTH;

          Vc::float_v signal = SAMPAProcessing::getGamma4(time, Vc::float_v(driftTime), Vc::float_v(nElectronsGEM*weight));
          Vc::float_v ADCsignal = SAMPAProcessing::getADCvalueVc(signal);
          Vc::float_m signalcondition = signal <= 0.f;

          for (int i=0;i<Vc::float_v::Size;++i) {
            if(signalcondition[i]) continue;
            mDigitContainer->addDigit(MCEventID, MCTrackID, digiPadPos.getCRU().number(), getTimeBinFromTime(time[i]), row, pad, ADCsignal[i]);
          }

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
