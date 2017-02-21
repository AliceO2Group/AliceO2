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

    /// Conversion from a given number of electrons into ADC value without taking into account saturation, vectorized
    /// @param nElectrons Number of electrons in time bin
    /// @return ADC value
    template<typename T>
    static T getADCvalue(T nElectrons);

    /// Saturation of the ADC
    /// @param signal Incoming signal in ADC counts
    /// @return ADC value
    template<typename T>
    static T getADCSaturation(T signal);

    static float getADCSaturation(float signal);

    /// Gamma4 shaping function, vectorized
    /// @param time Time of the ADC value with respect to the first bin in the pulse
    /// @param startTime First bin in the pulse
    /// @param ADC ADC value of the corresponding time bin
    template<typename T>
    static T getGamma4(T time, T startTime, T ADC);
};

template<typename T>
inline
T SAMPAProcessing::getADCvalue(T nElectrons)
{
  Vc::float_v conversion = QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE; // 1E-15 is to convert Coulomb in fC
  return nElectrons*conversion;
}

template<typename T>
inline
T SAMPAProcessing::getADCSaturation(T signal)
{
  Vc::float_v signalOut = signal;
  Vc::float_m saturation = signal > Vc::float_v(ADCSAT);
  signalOut(saturation) = Vc::float_v(ADCSAT);
  return signalOut;
}

inline
float SAMPAProcessing::getADCSaturation(float signal)
{
  if(signal > ADCSAT-1) signal = ADCSAT-1;
  return signal;
}

template<typename T>
inline
T SAMPAProcessing::getGamma4(T time, T startTime, T ADC)
{
  Vc::float_v tmp = (time-startTime)/PEAKINGTIME;
  Vc::float_v tmp2=tmp*tmp;
  return 55.f*ADC*Vc::exp(-4.f*tmp)*tmp2*tmp2; // 55 is for normalization: 1/Integral(Gamma4)
}
  
}
}

#endif // ALICEO2_TPC_SAMPAProcessing_H_
