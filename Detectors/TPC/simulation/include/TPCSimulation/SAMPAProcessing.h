/// \file SAMPAProcessing.h
/// \brief Definition of the SAMPA response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_SAMPAProcessing_H_
#define ALICEO2_TPC_SAMPAProcessing_H_

#include <Vc/Vc>

#include "TPCSimulation/Constants.h"

#include "TSpline.h"

namespace o2 {
namespace TPC {
    
/// \class SAMPAProcessing
/// This class takes care of the signal processing in the Front-End Cards (FECs), i.e. the shaping and the digitization.
/// Further effects such as saturation of the FECs are implemented.
    
class SAMPAProcessing
{
  public:
    static SAMPAProcessing& instance() {
      static SAMPAProcessing sampaProcessing;
      return sampaProcessing;
    }
    /// Destructor
    ~SAMPAProcessing();

    /// Conversion from a given number of electrons into ADC value without taking into account saturation (vectorized)
    /// \param nElectrons Number of electrons in time bin
    /// \return ADC value
    template<typename T>
    static T getADCvalue(T nElectrons);

    /// For larger input values the SAMPA response is not linear which is taken into account by this function
    /// \param signal Input signal
    /// \return ADC value of the (saturated) SAMPA
    const float getADCSaturation(const float signal) const;

    /// A delta signal is shaped by the FECs and thus spread over several time bins
    /// This function returns an array with the signal spread into the following time bins
    /// \param ADCsignal Signal of the incoming charge
    /// \param driftTime t0 of the incoming charge
    /// \return Array with the shaped signal
    static void getShapedSignal(float ADCsignal, float driftTime, std::array<float, mNShapedPoints> &signalArray);

    /// Value of the Gamma4 shaping function at a given time (vectorized)
    /// \param time Time of the ADC value with respect to the first bin in the pulse
    /// \param startTime First bin in the pulse
    /// \param ADC ADC value of the corresponding time bin
    template<typename T>
    static T getGamma4(T time, T startTime, T ADC);

  private:
    SAMPAProcessing();
    // use old c++03 due to root
    SAMPAProcessing(const SAMPAProcessing&) {}
    void operator=(const SAMPAProcessing&) {}

    std::unique_ptr<TSpline3>   mSaturationSpline;   ///< TSpline3 which holds the saturation curve

    /// Import the saturation curve from a .dat file to a TSpline3
    /// \param file Name of the .dat file
    /// \param spline TSpline3 to which the saturation curve will be written
    /// \return Boolean if succesful or not
    bool importSaturationCurve(std::string file);
};

template<typename T>
inline
T SAMPAProcessing::getADCvalue(T nElectrons)
{
  T conversion = QEL*1.e15*CHIPGAIN*ADCSAT/ADCDYNRANGE; // 1E-15 is to convert Coulomb in fC
  return nElectrons*conversion;
}

inline
const float SAMPAProcessing::getADCSaturation(const float signal) const
{
  /// \todo Performance of TSpline?
  const float saturatedSignal = mSaturationSpline->Eval(signal);
  if(saturatedSignal > ADCSAT-1) return ADCSAT-1;
  return saturatedSignal;
}

template<typename T>
inline
T SAMPAProcessing::getGamma4(T time, T startTime, T ADC)
{
  Vc::float_v tmp0 = (time-startTime)/PEAKINGTIME;
  Vc::float_m cond = (tmp0 > 0);
  Vc::float_v tmp;
  tmp(cond) = tmp0;
  Vc::float_v tmp2=tmp*tmp;
  return 55.f*ADC*Vc::exp(-4.f*tmp)*tmp2*tmp2; /// 55 is for normalization: 1/Integral(Gamma4)
}
}
}

#endif // ALICEO2_TPC_SAMPAProcessing_H_
