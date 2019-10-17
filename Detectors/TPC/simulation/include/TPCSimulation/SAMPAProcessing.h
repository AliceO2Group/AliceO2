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

#include "TPCBase/PadPos.h"
#include "TPCBase/CalDet.h"
#include "TPCBase/CRU.h"
#include "TPCBase/RandomRing.h"
#include "TPCBase/ParameterDetector.h"
#include "TPCBase/ParameterElectronics.h"
#include "TPCBase/ParameterGas.h"

#include "TSpline.h"

namespace o2
{
namespace tpc
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

  /// Make the full signal including noise and pedestals from the OCDB
  /// \param ADCcounts ADC value of the signal (common mode already subtracted)
  /// \param sector Sector number
  /// \param globalPadInSector global pad number in the sector
  /// \param commonMode value of the common mode
  /// \return ADC value after application of noise, pedestal and saturation
  template <DigitzationMode MODE>
  float makeSignal(float ADCcounts, const int sector, const int globalPadInSector, const float commonMode, float& pedestal, float& noise);

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

  /// Get the noise for a given channel
  /// \param cru CRU of the channel of interest
  /// \param padPos PadPos of the channel of interest
  /// \return Noise on the channel of interest
  float getNoise(const int sector, const int globalPadInSector);

  /// Get the pedestal for a given channel
  /// \param cru CRU of the channel of interest
  /// \param padPos PadPos of the channel of interest
  /// \return Pedestal on the channel of interest
  float getPedestal(const int sector, const int globalPadInSector) const;

 private:
  SAMPAProcessing();

  const ParameterGas* mGasParam;         ///< Caching of the parameter class to avoid multiple CDB calls
  const ParameterDetector* mDetParam;    ///< Caching of the parameter class to avoid multiple CDB calls
  const ParameterElectronics* mEleParam; ///< Caching of the parameter class to avoid multiple CDB calls
  const CalPad* mNoiseMap;               ///< Caching of the parameter class to avoid multiple CDB calls
  const CalPad* mPedestalMap;            ///< Caching of the parameter class to avoid multiple CDB calls
  RandomRing<> mRandomNoiseRing;         ///< Ring with random number for noise
};

template <typename T>
inline T SAMPAProcessing::getADCvalue(T nElectrons) const
{
  T conversion = mEleParam->ElectronCharge * 1.e15 * mEleParam->ChipGain * mEleParam->ADCsaturation /
                 mEleParam->ADCdynamicRange; // 1E-15 is to convert Coulomb in fC
  return nElectrons * conversion;
}

template <DigitzationMode MODE>
inline float SAMPAProcessing::makeSignal(float ADCcounts, const int sector, const int globalPadInSector, const float commonMode,
                                         float& pedestal, float& noise)
{
  float signal = ADCcounts;
  pedestal = getPedestal(sector, globalPadInSector);
  noise = getNoise(sector, globalPadInSector);
  switch (MODE) {
    case DigitzationMode::FullMode: {
      signal -= commonMode;
      signal += noise;
      signal += pedestal;
      return getADCSaturation(signal);
      break;
    }
    case DigitzationMode::SubtractPedestal: {
      signal -= commonMode;
      signal += noise;
      signal += pedestal;
      float signalSubtractPedestal = getADCSaturation(signal) - pedestal;
      return signalSubtractPedestal;
      break;
    }
    case DigitzationMode::NoSaturation: {
      signal -= commonMode;
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
  return signal;
}

inline float SAMPAProcessing::getADCSaturation(const float signal) const
{
  const float adcSaturation = mEleParam->ADCsaturation;
  if (signal > adcSaturation - 1)
    return adcSaturation - 1;
  return signal;
}

template <typename T>
inline T SAMPAProcessing::getGamma4(T time, T startTime, T ADC) const
{
  const auto tmp0 = (time - startTime) / mEleParam->PeakingTime;
  const auto cond = (tmp0 > 0);
  T tmp{};
  if constexpr (std::is_floating_point_v<T>) {
    if (!cond) {
      return T{};
    }
    tmp = tmp0;
  } else {
    tmp(cond) = tmp0;
  }
  const auto tmp2 = tmp * tmp;
  return 55.f * ADC * std::exp(-4.f * tmp) * tmp2 * tmp2; /// 55 is for normalization: 1/Integral(Gamma4)
}

inline TimeBin SAMPAProcessing::getTimeBin(float zPos) const
{
  float timeBin = (mDetParam->TPClength - std::abs(zPos)) / (mGasParam->DriftV * mEleParam->ZbinWidth);
  return static_cast<TimeBin>(timeBin);
}

inline float SAMPAProcessing::getZfromTimeBin(float timeBin, Side s) const
{
  float zSign = (s == 0) ? 1 : -1;
  float zAbs = zSign * (mDetParam->TPClength - (timeBin * mGasParam->DriftV * mEleParam->ZbinWidth));
  return zAbs;
}

inline TimeBin SAMPAProcessing::getTimeBinFromTime(float time) const
{
  float timeBin = time / mEleParam->ZbinWidth;
  return static_cast<TimeBin>(timeBin);
}

inline float SAMPAProcessing::getTimeFromBin(TimeBin timeBin) const
{
  float time = static_cast<float>(timeBin) * mEleParam->ZbinWidth;
  return time;
}

inline float SAMPAProcessing::getTimeBinTime(float time) const
{
  TimeBin timeBin = getTimeBinFromTime(time);
  return getTimeFromBin(timeBin);
}

inline float SAMPAProcessing::getNoise(const int sector, const int globalPadInSector)
{
  return mRandomNoiseRing.getNextValue() * mNoiseMap->getValue(sector, globalPadInSector);
}

inline float SAMPAProcessing::getPedestal(const int sector, const int globalPadInSector) const
{
  return mPedestalMap->getValue(sector, globalPadInSector);
}
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_SAMPAProcessing_H_
