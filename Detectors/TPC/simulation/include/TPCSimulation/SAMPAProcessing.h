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

namespace o2
{
namespace TPC
{

/// \class SAMPAProcessing
/// This class takes care of the signal processing in the Front-End Cards (FECs), i.e. the shaping and the digitization
/// Further effects such as saturation of the FECs are implemented.

class SAMPAProcessing
{
 public:
  static SAMPAProcessing& instance()
  {
    static SAMPAProcessing sampaProcessing;
    return sampaProcessing;
  }
  /// Destructor
  ~SAMPAProcessing();

  /// Update the OCDB parameters cached in the class. To be called once per event
  void updateParameters();

  /// Conversion from a given number of electrons into ADC value without taking into account saturation (vectorized)
  /// \param nElectrons Number of electrons in time bin
  /// \return ADC value
  template <typename T>
  T getADCvalue(T nElectrons) const;

  /// For larger input values the SAMPA response is not linear which is taken into account by this function
  /// \param signal Input signal
  /// \return ADC value of the (saturated) SAMPA
  float getADCSaturation(const float signal) const;

  /// Make the full signal including noise and pedestals from the Baseline class
  /// \param ADCcounts ADC value of the signal (common mode already subtracted)
  /// \param padSecPos PadSecPos of the signal
  /// \return ADC value after application of noise, pedestal and saturation
  float makeSignal(float ADCcounts, const PadSecPos& padSecPos, float& pedestal, float& noise) const;

  /// A delta signal is shaped by the FECs and thus spread over several time bins
  /// This function returns an array with the signal spread into the following time bins
  /// \param ADCsignal Signal of the incoming charge
  /// \param driftTime t0 of the incoming charge
  /// \return Array with the shaped signal
  /// \todo the size of the array should be retrieved from ParameterElectronics::getNShapedPoints()
  void getShapedSignal(float ADCsignal, float driftTime, std::vector<float>& signalArray) const;

  /// Value of the Gamma4 shaping function at a given time (vectorized)
  /// \param time Time of the ADC value with respect to the first bin in the pulse
  /// \param startTime First bin in the pulse
  /// \param ADC ADC value of the corresponding time bin
  template <typename T>
  T getGamma4(T time, T startTime, T ADC) const;

  /// Compute time bin from z position
  /// \param zPos z position of the charge
  /// \return Time bin of the charge
  TimeBin getTimeBin(float zPos) const;

  /// Compute z position from time bin
  /// \param Time bin of the charge
  /// \param Side of the TPC
  /// \return zPos z position of the charge
  float getZfromTimeBin(float timeBin, Side s) const;

  /// Compute time bin from time
  /// \param time time of the charge
  /// \return Time bin of the charge
  TimeBin getTimeBinFromTime(float time) const;

  /// Compute time from time bin
  /// \param timeBin time bin of the charge
  /// \return Time of the charge
  float getTimeFromBin(TimeBin timeBin) const;

  /// Compute the time of a given time bin
  /// \param time Time of the charge
  /// \return Time of the time bin of the charge
  float getTimeBinTime(float time) const;

 private:
  SAMPAProcessing();

  std::unique_ptr<TSpline3> mSaturationSpline; ///< TSpline3 which holds the saturation curve

  /// Import the saturation curve from a .dat file to a TSpline3
  /// \param file Name of the .dat file
  /// \param spline TSpline3 to which the saturation curve will be written
  /// \return Boolean if succesful or not
  bool importSaturationCurve(std::string file);

  const ParameterGas* mGasParam;         ///< Caching of the parameter class to avoid multiple CDB calls
  const ParameterDetector* mDetParam;    ///< Caching of the parameter class to avoid multiple CDB calls
  const ParameterElectronics* mEleParam; ///< Caching of the parameter class to avoid multiple CDB calls
};

template <typename T>
inline T SAMPAProcessing::getADCvalue(T nElectrons) const
{
  T conversion = mEleParam->getElectronCharge() * 1.e15 * mEleParam->getChipGain() * mEleParam->getADCSaturation() /
                 mEleParam->getADCDynamicRange(); // 1E-15 is to convert Coulomb in fC
  return nElectrons * conversion;
}

inline float SAMPAProcessing::makeSignal(float ADCcounts, const PadSecPos& padSecPos,
                                         float& pedestal, float& noise) const
{
  static Baseline baseline;
  float signal = ADCcounts;
  /// \todo Pedestal to be implemented in baseline class
  //  pedestal = baseline.getPedestal(padSecPos);
  noise = baseline.getNoise(padSecPos);
  switch (mEleParam->getDigitizationMode()) {
    case DigitzationMode::FullMode: {
      signal += noise;
      signal += pedestal;
      return getADCSaturation(signal);
      break;
    }
    case DigitzationMode::SubtractPedestal: {
      signal += noise;
      signal += pedestal;
      float signalSubtractPedestal = getADCSaturation(signal) - pedestal;
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

inline float SAMPAProcessing::getADCSaturation(const float signal) const
{
  /// \todo Performance of TSpline?
  const float saturatedSignal = mSaturationSpline->Eval(signal);
  const float adcSaturation = mEleParam->getADCSaturation();
  if (saturatedSignal > adcSaturation - 1)
    return adcSaturation - 1;
  return saturatedSignal;
}

template <typename T>
inline T SAMPAProcessing::getGamma4(T time, T startTime, T ADC) const
{
  Vc::float_v tmp0 = (time - startTime) / mEleParam->getPeakingTime();
  Vc::float_m cond = (tmp0 > 0);
  Vc::float_v tmp;
  tmp(cond) = tmp0;
  Vc::float_v tmp2 = tmp * tmp;
  return 55.f * ADC * Vc::exp(-4.f * tmp) * tmp2 * tmp2; /// 55 is for normalization: 1/Integral(Gamma4)
}

inline TimeBin SAMPAProcessing::getTimeBin(float zPos) const
{
  float timeBin = (mDetParam->getTPClength() - std::abs(zPos)) / (mGasParam->getVdrift() * mEleParam->getZBinWidth());
  return static_cast<TimeBin>(timeBin);
}

inline float SAMPAProcessing::getZfromTimeBin(float timeBin, Side s) const
{
  float zSign = (s == 0) ? 1 : -1;
  float zAbs = zSign * (mDetParam->getTPClength() - (timeBin * mGasParam->getVdrift() * mEleParam->getZBinWidth()));
  return zAbs;
}

inline TimeBin SAMPAProcessing::getTimeBinFromTime(float time) const
{
  float timeBin = time / mEleParam->getZBinWidth();
  return static_cast<TimeBin>(timeBin);
}

inline float SAMPAProcessing::getTimeFromBin(TimeBin timeBin) const
{
  float time = static_cast<float>(timeBin) * mEleParam->getZBinWidth();
  return time;
}

inline float SAMPAProcessing::getTimeBinTime(float time) const
{
  TimeBin timeBin = getTimeBinFromTime(time);
  return getTimeFromBin(timeBin);
}
}
}

#endif // ALICEO2_TPC_SAMPAProcessing_H_
