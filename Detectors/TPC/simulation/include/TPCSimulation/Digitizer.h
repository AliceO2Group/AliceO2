/// \file Digitizer.h
/// \brief Task for ALICE TPC digitization
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_Digitizer_H_
#define ALICEO2_TPC_Digitizer_H_

#include "TPCSimulation/DigitContainer.h"
#include "TPCSimulation/PadResponse.h"
#include "TPCSimulation/Constants.h"

#include "TPCBase/RandomRing.h"

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
    void ElectronDrift(Float_t *xyz);

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
    Float_t ADCvalue(Float_t nElectrons) const;

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
    Float_t Gamma4(Float_t time, Float_t startTime, Float_t ADC) const;

  private:
    Digitizer(const Digitizer &);
    Digitizer &operator=(const Digitizer &);

    DigitContainer          *mDigitContainer;
      
    RandomRing              mRandomGaus;
    
  ClassDef(Digitizer, 1);
};

// inline implementations

inline
Float_t Digitizer::ADCvalue(Float_t nElectrons) const
{
  Float_t adcValue = nElectrons*QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE;
  if(adcValue > ADCSAT) adcValue = ADCSAT;// saturation
  
  return adcValue;
}

inline
Int_t Digitizer::getTimeBin(Float_t zPos) const 
{
  Float_t timeBin = (TPCLENGTH-TMath::Abs(zPos))/(DRIFTV*ZBINWIDTH);
  return static_cast<int>(timeBin);
}

inline
Int_t Digitizer::getTimeBinFromTime(Float_t time) const 
{
  Float_t timeBin = time / ZBINWIDTH;
  return static_cast<int>(timeBin);
}

inline
Float_t Digitizer::getTimeFromBin(Int_t timeBin) const 
{
  Float_t time = static_cast<float>(timeBin)*ZBINWIDTH;
  return time;
}

inline
Float_t Digitizer::getTime(Float_t zPos) const 
{
  Float_t time = (TPCLENGTH-TMath::Abs(zPos))/DRIFTV;
  return time;
}

inline
Float_t Digitizer::Gamma4(Float_t time, Float_t startTime, Float_t ADC) const 
{
  if (time<0) return 0;
  return ADC*TMath::Exp(-4.*(time-startTime)/PEAKINGTIME)*TMath::Power((time-startTime)/PEAKINGTIME,4);
}

inline
void Digitizer::getPadResponse(Float_t xabs, Float_t yabs, std::vector<PadResponse> &response)
{
  response.resize(0);
  /// @todo include actual response, this is now only for a signal on the central pad (0, 0) with weight 1.
  response.emplace_back(0, 0, 1);
}

}
}

#endif // ALICEO2_TPC_Digitizer_H_
