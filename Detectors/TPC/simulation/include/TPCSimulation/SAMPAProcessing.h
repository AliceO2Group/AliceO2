/// \file SAMPAProcessing.h
/// \brief This class handles the shaping and digitization on the FECs
/// \author Andi Mathis, andreas.mathis@ph.tum.de
#ifndef ALICEO2_TPC_SAMPAProcessing_H_
#define ALICEO2_TPC_SAMPAProcessing_H_

#include <Vc/Vc>

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
    static float getADCvalue(float nElectrons);

    /// Conversion from a given number of electrons into ADC value without taking into account saturation, vectorized
    /// @param nElectrons Number of electrons in time bin
    /// @return ADC value
    static Vc::float_v getADCvalueVc(Vc::float_v nElectrons);

    /// Saturation of the ADC
    /// @param signal Incoming signal in ADC counts
    /// @return ADC value
    static float getADCSaturation(float signal);

    /// Gamma4 shaping function
    /// @param time Time of the ADC value with respect to the first bin in the pulse
    /// @param startTime First bin in the pulse
    /// @param ADC ADC value of the corresponding time bin
//     static float getGamma4(float time, float startTime, float ADC);

    /// Gamma4 shaping function, vectorized
    /// @param time Time of the ADC value with respect to the first bin in the pulse
    /// @param startTime First bin in the pulse
    /// @param ADC ADC value of the corresponding time bin
//     __attribute__((noinline))
    template<typename T>
    static T getGamma4(T time, T startTime, T ADC);
};

inline
float SAMPAProcessing::getADCvalue(float nElectrons)
{
  float adcValue = nElectrons*QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE;
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
float SAMPAProcessing::getADCSaturation(float signal)
{
  if(signal > ADCSAT) signal = ADCSAT;
  return signal;
}

// inline
// float SAMPAProcessing::getGamma4(float time, float startTime, float ADC)
// {
// //   if (time<0) return 0;
//   float_t tmp = (time-startTime)/PEAKINGTIME;
//   float_t tmp2=tmp*tmp;
//   return 55.f*ADC*exp(-4.f*tmp)*tmp2*tmp2;
// }

// inline
template<typename T>
__attribute__((noinline))
T SAMPAProcessing::getGamma4(T time, T startTime, T ADC)
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
