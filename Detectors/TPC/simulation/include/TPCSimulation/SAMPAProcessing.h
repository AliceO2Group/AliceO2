// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SAMPAProcessing.h
/// \brief Definition of the SAMPA response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_SAMPAProcessing_H_
#define ALICEO2_TPC_SAMPAProcessing_H_

#include <Vc/Vc>

#include "TPCBase/PadSecPos.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"
#include "TPCSimulation/Baseline.h"

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

    /// Make the full signal including noise and pedestals from the Baseline class
    /// \param ADCcounts ADC value of the signal (common mode already subtracted)
    /// \param padSecPos PadSecPos of the signal
    /// \return ADC value after application of noise, pedestal and saturation
    static float makeSignal(float ADCcounts, const PadSecPos &padSecPos, float &pedestal, float &noise);

    /// A delta signal is shaped by the FECs and thus spread over several time bins
    /// This function returns an array with the signal spread into the following time bins
    /// \param ADCsignal Signal of the incoming charge
    /// \param driftTime t0 of the incoming charge
    /// \return Array with the shaped signal
    /// \todo the size of the array should be retrieved from ParameterElectronics::getNShapedPoints()
    static void getShapedSignal(float ADCsignal, float driftTime, std::vector<float> &signalArray);

    /// Value of the Gamma4 shaping function at a given time (vectorized)
    /// \param time Time of the ADC value with respect to the first bin in the pulse
    /// \param startTime First bin in the pulse
    /// \param ADC ADC value of the corresponding time bin
    template<typename T>
    static T getGamma4(T time, T startTime, T ADC);

    /// Compute time bin from z position
    /// \param zPos z position of the charge
    /// \return Time bin of the charge
    static TimeBin getTimeBin(float zPos);

    /// Compute z position from time bin
    /// \param Time bin of the charge
    /// \param Side of the TPC
    /// \return zPos z position of the charge
    static float getZfromTimeBin(float timeBin, Side s);

    /// Compute time bin from time
    /// \param time time of the charge
    /// \return Time bin of the charge
    static TimeBin getTimeBinFromTime(float time);

    /// Compute time from time bin
    /// \param timeBin time bin of the charge
    /// \return Time of the charge
    static float getTimeFromBin(TimeBin timeBin);

    /// Compute electron drift time from z position
    /// \param zPos z position of the charge
    /// \return Time of the charge
    static float getDriftTime(float zPos);

    /// Compute the time of a given time bin
    /// \param time Time of the charge
    /// \return Time of the time bin of the charge
    static float getTimeBinTime(float time);

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
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  T conversion = eleParam.getElectronCharge()*1.e15*eleParam.getChipGain()*eleParam.getADCSaturation()/eleParam.getADCDynamicRange(); // 1E-15 is to convert Coulomb in fC
  return nElectrons*conversion;
}

inline
float SAMPAProcessing::makeSignal(float ADCcounts, const PadSecPos& padSecPos, float &pedestal, float &noise)
{
  SAMPAProcessing &sampa = SAMPAProcessing::instance();
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  static Baseline baseline;
  float signal = ADCcounts;
  /// \todo Pedestal to be implemented in baseline class
//  pedestal = baseline.getPedestal(padSecPos);
  noise = baseline.getNoise(padSecPos);
  switch (eleParam.getDigitizationMode()) {
    case DigitzationMode::FullMode: {
      signal += noise;
      signal += pedestal;
      return sampa.getADCSaturation(signal);
      break;
    }
    case DigitzationMode::SubtractPedestal: {
      signal += noise;
      signal += pedestal;
      float signalSubtractPedestal = sampa.getADCSaturation(signal) - pedestal;
      return signalSubtractPedestal;
      break;
    }
    case DigitzationMode::NoSaturation: {
      signal += noise;
      signal += pedestal;
      return signal;
      break;
    }
    case DigitzationMode::PropagateADC: {
      return signal;
      break;
    }
  }
}

inline
const float SAMPAProcessing::getADCSaturation(const float signal) const
{
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  /// \todo Performance of TSpline?
  const float saturatedSignal = mSaturationSpline->Eval(signal);
  const float adcSaturation = eleParam.getADCSaturation();
  if(saturatedSignal > adcSaturation-1) return adcSaturation-1;
  return saturatedSignal;
}

template<typename T>
inline
T SAMPAProcessing::getGamma4(T time, T startTime, T ADC)
{
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  Vc::float_v tmp0 = (time-startTime)/eleParam.getPeakingTime();
  Vc::float_m cond = (tmp0 > 0);
  Vc::float_v tmp;
  tmp(cond) = tmp0;
  Vc::float_v tmp2=tmp*tmp;
  return 55.f*ADC*Vc::exp(-4.f*tmp)*tmp2*tmp2; /// 55 is for normalization: 1/Integral(Gamma4)
}

inline
TimeBin SAMPAProcessing::getTimeBin(float zPos)
{
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  float timeBin = (detParam.getTPClength()-std::abs(zPos))/(gasParam.getVdrift()*eleParam.getZBinWidth());
  return static_cast<int>(timeBin);
}

inline
float SAMPAProcessing::getZfromTimeBin(float timeBin, Side s)
{
  float zSign = (s==0) ? 1 : -1;
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  float zAbs =  zSign * (detParam.getTPClength()- (timeBin*gasParam.getVdrift()*eleParam.getZBinWidth()));
  return zAbs;
}

inline
TimeBin SAMPAProcessing::getTimeBinFromTime(float time)
{
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  float timeBin = time / eleParam.getZBinWidth();
  return static_cast<int>(timeBin);
}

inline
float SAMPAProcessing::getTimeFromBin(TimeBin timeBin)
{
  const static ParameterElectronics &eleParam = ParameterElectronics::defaultInstance();
  float time = static_cast<float>(timeBin)*eleParam.getZBinWidth();
  return time;
}

inline
float SAMPAProcessing::getDriftTime(float zPos)
{
  const static ParameterGas &gasParam = ParameterGas::defaultInstance();
  const static ParameterDetector &detParam = ParameterDetector::defaultInstance();
  float time = (detParam.getTPClength()-std::abs(zPos))/gasParam.getVdrift();
  return time;
}

inline
float SAMPAProcessing::getTimeBinTime(float time)
{
  TimeBin timeBin = getTimeBinFromTime(time);
  return getTimeFromBin(timeBin);
}

}
}

#endif // ALICEO2_TPC_SAMPAProcessing_H_
