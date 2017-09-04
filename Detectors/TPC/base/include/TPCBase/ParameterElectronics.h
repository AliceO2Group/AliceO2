// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ParameterElectronics.h
/// \brief Definition of the parameter class for the detector electronics
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_ParameterElectronics_H_
#define ALICEO2_TPC_ParameterElectronics_H_

#include <array>

namespace o2 {
namespace TPC {

/// \class ParameterElectronics

class ParameterElectronics{
  public:
    static ParameterElectronics& defaultInstance() {
      static ParameterElectronics param;
      param.setDefaultValues();
      return param;
    }

    /// Constructor
    ParameterElectronics();

    /// Destructor
    ~ParameterElectronics() = default;

    /// Set the default values
    void setDefaultValues();

    /// Set number of ADC samples with which are taken into account for a given, shaped signal
    /// \param nShaped Number of shaped ADC samples
    void setNShapedPoints(int nShaped) { mNShapedPoints = nShaped; }

    /// Set SAMPA peaking time
    /// \param peakingTime SAMPA peaking time [us]
    void setPeakingTime(float peakingTime) { mPeakingTime = peakingTime; }

    /// Set SAMPA chip gain
    /// \param chipGain SAMPA chip gain [mV/fC]
    void setChipGain(float chipGain) { mChipGain = chipGain; }

    /// Set ADC dynamic range
    /// \param dynRange ADC dynamic range [mV]
    void setADCDynamicRange(float dynRange) { mADCdynamicRange = dynRange; }

    /// Set ADC saturation
    /// \param adcSat ADC saturation [ADC counts]
    void setADCSaturation(float adcSat) { mADCsaturation = adcSat; }

    /// Set z-bin width
    /// \param zbin z-bin width [us]
    void setZBinWidth(float zbin) { mZbinWidth = zbin; }

    /// Set electron charge
    /// \param qel Electron charge [C]
    void setElectronCharge(float qel) { mElectronCharge = qel; }


    /// Get number of ADC samples which are taken into account for a given, shaped signal
    /// \return Number of shaped ADC samples
    int getNShapedPoints() const { return mNShapedPoints; }

    /// Get SAMPA peaking time
    /// \return SAMPA peaking time [us]
    float getPeakingTime() const { return mPeakingTime; }

    /// Get SAMPA chip gain
    /// \return SAMPA chip gain [mV/fC]
    float getChipGain() const { return mChipGain; }

    /// Get ADC dynamic range
    /// \return ADC dynamic range [mV]
    float getADCDynamicRange() const { return mADCdynamicRange; }

    /// Get ADC saturation
    /// \return ADC saturation [ADC counts]
    float getADCSaturation() const { return mADCsaturation; }

    /// Get z-bin width
    /// \return z-bin width [us]
    float getZBinWidth() const { return mZbinWidth; }

    /// Get electron charge
    /// \return Electron charge [C]
    float getElectronCharge() const { return mElectronCharge; }


  private:

    int mNShapedPoints;                    ///< Number of ADC samples which are taken into account for a given, shaped signal (should fit into SSE registers)
    float mPeakingTime;                    ///< Peaking time of the SAMPA [us]
    float mChipGain;                       ///< Gain of the SAMPA [mV/fC] - may be either 20 or 30
    float mADCdynamicRange;                ///< Dynamic range of the ADC [mV]
    float mADCsaturation;                  ///< ADC saturation [ADC counts]
    float mZbinWidth;                      ///< Width of a z bin [us]
    float mElectronCharge;                 ///< Electron charge [C]
  };

}
}

#endif // ALICEO2_TPC_ParameterElectronics_H_
