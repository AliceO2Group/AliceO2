/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"

#include "Rtypes.h"
#include "TObject.h"
#include "TF1.h"
#include "TRandom.h"
#include "TMath.h"
#include <iostream>

using std::vector;

class TClonesArray;

namespace AliceO2{
  namespace TPC{
    
    class DigitContainer;
    
    /// \class Digitizer
    /// \brief Digitizer class for the TPC
    
    class Digitizer : public TObject {
    public:
      
      /// Default constructor
      Digitizer();
      
      /// Destructor
      ~Digitizer();
      
      /// Initializer
      void init();
      
      /// Steer conversion of points to digits
      /// @param points Container with TPC points
      /// @return digits container
      DigitContainer *Process(TClonesArray *points);
      
      /// Drift of electrons in electric field taking into account diffusion
      /// @param *xyz Array with 3d position of the electrons
      /// @return Array with 3d position of the electrons after the drift taking into account diffusion
      void ElectronDrift(Float_t *xyz) const;
      
      /// Simulation of the GEM response
      /// @return Number of electrons after GEM amplification taking into account exponential fluctuations of the gain
      Float_t GEMAmplification() const;
      
      
      /// Simulation of the GEM response of a single GEM
      /// @param nEle Number of incoming electrons
      /// @param gain Gain of the single GEM, should go to OCDB
      /// @return Number of electrons after GEM amplification taking into account fluctuations of the gain according to a Polya distribution
      Int_t SingleGEMAmplification(Int_t nEle, Float_t gain) const;
      
      
      /// Pad Response
      /// @param xabs Position in x
      /// @param yabs Position in y
      /// @return Vector with PadResponse objects with pad and row position and the correponding fraction of the induced signal
      void getPadResponse(Float_t xabs, Float_t yabs, vector<PadResponse> &);
      
      
      /// ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
      /// Conversion functions that at some point should go someplace else
      
      /// Conversion from a given number of electrons into ADC value
      /// @param nElectrons Number of electrons in time bin
      /// @return ADC value
      Int_t ADCvalue(Float_t nElectrons) const;
      
      /// Compute time bin from z position
      /// @param zPos z position of the charge
      /// @return Time bin of the charge
      Int_t getTimeBin(Float_t zPos) const;
      
      /// Compute time bin from time
      /// @param time time of the charge
      /// @return Time bin of the charge
      Int_t getTimeBinFromTime(Float_t time) const;
      
      /// Compute time from time bin
      /// @param timeBin time bin of the charge
      /// @return Time of the charge      
      Float_t getTimeFromBin(Int_t timeBin) const;
      
      /// Compute time from z position
      /// @param zPos z position of the charge
      /// @return Time of the charge
      Float_t getTime(Float_t zPos) const;
      
      /// Gamma4 shaping function
      /// @param time Time of the ADC value with respect to the first bin in the pulse
      /// @param startTime First bin in the pulse
      /// @param ADC ADC value of the corresponding time bin
      /// TODO: Could we live with Float_t ? (Question SW)
      Float_t Gamma4(Float_t time, Float_t startTime, Float_t ADC) const;
      
    private:
      Digitizer(const Digitizer &);
      Digitizer &operator=(const Digitizer &);
      
      TF1                     *mPolya;
      DigitContainer          *mDigitContainer;
      std::vector<PadResponse> mPadResponse;
      ClassDef(Digitizer, 1);
    };
    
    // inline implementations
    inline
    Float_t Digitizer::GEMAmplification() const {
      // TODO parameters to be stored someplace else
      Float_t gain = 2000;
      
      Float_t rn=TMath::Max(gRandom->Rndm(0),1.93e-22);
      Float_t signal = static_cast<int>(-(gain) * TMath::Log(rn));
      return signal;
    }
    
    inline
    Int_t Digitizer::SingleGEMAmplification(Int_t nEle, Float_t gain) const {
      //   return static_cast<int>(static_cast<float>(nEle)*gain*mPolya->GetRandom());
      return static_cast<int>(static_cast<float>(nEle)*gain);
    }
    
    inline
    Int_t Digitizer::ADCvalue(Float_t nElectrons) const {
      // TODO parameters to be stored someplace else
      Float_t ADCSat = 1023;
      Float_t Qel = 1.602e-19;
      Float_t ChipGain = 20;
      Float_t ADCDynRange = 2000;
      
      Int_t adcValue = static_cast<int>(nElectrons*Qel*1.e15*ChipGain*ADCSat/ADCDynRange);
      if(adcValue > ADCSat) adcValue = ADCSat;// saturation
      
      return adcValue;
    }
    
    inline
    Int_t Digitizer::getTimeBin(Float_t zPos) const {
      // TODO parameters to be stored someplace else
      Float_t driftV = 2.58;
      Float_t zBinWidth = 0.19379844961;
      
      Float_t timeBin = (250.-TMath::Abs(zPos))/(driftV*zBinWidth);
      return static_cast<int>(timeBin);
    }
    
    inline
    Int_t Digitizer::getTimeBinFromTime(Float_t time) const {
      // TODO parameters to be stored someplace else
      Float_t zBinWidth = 0.19379844961;
      
      Float_t timeBin = time / zBinWidth;
      return static_cast<int>(timeBin);
    }
    
    inline
    Float_t Digitizer::getTimeFromBin(Int_t timeBin) const {
      // TODO parameters to be stored someplace else
      Float_t zBinWidth = 0.19379844961;
      
      Float_t time = static_cast<float>(timeBin)*zBinWidth;
      return time;
    }
    
    inline
    Float_t Digitizer::getTime(Float_t zPos) const {
      // TODO parameters to be stored someplace else
      Float_t driftV = 2.58;
      
      Float_t time = (250.-TMath::Abs(zPos))/driftV;
      return time;
    }
    
    inline
    Float_t Digitizer::Gamma4(Float_t time, Float_t startTime, Float_t ADC) const {
      // TODO parameters to be stored someplace else
      Float_t peakingTime = 160e-3; // all times are in us
      
      if (time<0) return 0;
      return ADC*TMath::Exp(-4.*(time-startTime)/peakingTime)*TMath::Power((time-startTime)/peakingTime,4);
    }
    
    inline
    void Digitizer::getPadResponse(Float_t xabs, Float_t yabs, std::vector<PadResponse> &response){
      response.resize(0);
      response.emplace_back(0, 0, 1);
    }
    
  }
}

#endif /* ALICEO2_TPC_Digitizer_H_ */