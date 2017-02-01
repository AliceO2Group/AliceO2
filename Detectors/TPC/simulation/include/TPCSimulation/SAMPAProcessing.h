/// \file SAMPAProcessing.h
/// \brief This class handles the shaping and digitization on the FECs
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_SAMPAProcessing_H_
#define ALICEO2_TPC_SAMPAProcessing_H_

#include <Vc/Vc>

#include "Rtypes.h"
#include "TPCSimulation/Constants.h"

namespace AliceO2 {
namespace TPC {
    
/// \class SAMPAProcessing
/// \brief Class taking care of the signal processing in the FEC, i.e. the shaping and the digitization
    
class SAMPAProcessing
{
  public:

    /// Default constructor
    SAMPAProcessing();

    /// Destructor
    ~SAMPAProcessing();

    /// Conversion from a given number of electrons into ADC value without taking into account saturation
    /// @param nElectrons Number of electrons in time bin
    /// @return ADC value
    static Float_t getADCvalue(Float_t nElectrons);

    /// Conversion from a given number of electrons into ADC value without taking into account saturation, vectorized
    /// @param nElectrons Number of electrons in time bin
    /// @return ADC value
    static Vc::float_v getADCvalueVc(Vc::float_v nElectrons);

    /// Saturation of the ADC
    /// @param signal Incoming signal in ADC counts
    /// @return ADC value
    static Float_t getADCSaturation(Float_t signal);

    /// Gamma4 shaping function
    /// @param time Time of the ADC value with respect to the first bin in the pulse
    /// @param startTime First bin in the pulse
    /// @param ADC ADC value of the corresponding time bin
    static Float_t getGamma4(Float_t time, Float_t startTime, Float_t ADC);

    /// Gamma4 shaping function, vectorized
    /// @param time Time of the ADC value with respect to the first bin in the pulse
    /// @param startTime First bin in the pulse
    /// @param ADC ADC value of the corresponding time bin
    static Vc::float_v getGamma4Vc(Vc::float_v time, Vc::float_v startTime, Vc::float_v ADC);
};

inline
Float_t SAMPAProcessing::getADCvalue(Float_t nElectrons)
{
  Float_t adcValue = nElectrons*QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE;
  return adcValue;
}

inline
Vc::float_v SAMPAProcessing::getADCvalueVc(Vc::float_v nElectrons)
{
  Vc::float_v tmp = QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE;
  Vc::float_v adcValue = nElectrons*tmp;
  return adcValue;
}

inline
Float_t SAMPAProcessing::getADCSaturation(Float_t signal)
{
  if(signal > ADCSAT) signal = ADCSAT;
  return signal;
}

inline
Float_t SAMPAProcessing::getGamma4(Float_t time, Float_t startTime, Float_t ADC)
{
//   if (time<0) return 0;
  float_t tmp = (time-startTime)/PEAKINGTIME;
  float_t tmp2=tmp*tmp;
  return 55.f*ADC*std::exp(-4.f*tmp)*tmp2*tmp2;
}

inline
Vc::float_v SAMPAProcessing::getGamma4Vc(Vc::float_v time, Vc::float_v startTime, Vc::float_v ADC)
{
  // not doing if because disregarded later in digitization
  // if (time<0) return 0;
  Vc::float_v tmp = (time-startTime)/PEAKINGTIME;
  Vc::float_v tmp2=tmp*tmp;
  return 55.f*ADC*Vc::exp(-4.f*tmp)*tmp2*tmp2;
}
  
}
}

#endif // ALICEO2_TPC_SAMPAProcessing_H_
