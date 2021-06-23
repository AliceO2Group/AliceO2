// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file DigitMCMetaData.h
/// \brief Definition of the Meta Data object of the Monte Carlo Digit
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_DigitMCMetaData_H_
#define ALICEO2_TPC_DigitMCMetaData_H_

#include <Rtypes.h>

namespace o2
{
namespace tpc
{

/// \class DigitMCMetaData
/// This is the definition of the Meta Data object of the Monte Carlo Digit
/// It holds auxilliary information relevant for debugging:
/// ADC value, Common Mode, Pedestal and Noise

class DigitMCMetaData
{
 public:
  /// Default constructor
  DigitMCMetaData() = default;

  /// Constructor, initializing values for position, charge, time and common mode
  /// \param ADC Raw ADC value of the corresponding DigitMC (before saturation)
  /// \param commonMode Common mode signal on that ROC in the time bin of the DigitMC
  /// \param pedestal Raw pedestal value of the corresponding DigitMC (before saturation)
  /// \param noise Raw noise value of the corresponding DigitMC (before saturation)
  DigitMCMetaData(float ADC, float commonMode, float pedestal, float noise);

  /// Destructor
  ~DigitMCMetaData() = default;

  /// Get the raw ADC value
  /// \return Raw ADC value of the corresponding DigitMC (before saturation)
  float getRawADC() const { return mADC; }

  /// Get the common mode value
  /// \return Common mode signal on that ROC in the time bin of the DigitMC
  float getCommonMode() const { return mCommonMode; }

  /// Get the raw pedestal value
  /// \return Raw pedestal value of the corresponding DigitMC (before saturation)
  float getPedestal() const { return mPedestal; }

  /// Get the raw noise value
  /// \return Raw noise value of the corresponding DigitMC (before saturation)
  float getNoise() const { return mNoise; }

 private:
  float mADC = 0.f;        ///< Raw ADC value of the DigitMCMetaData
  float mCommonMode = 0.f; ///< Common mode value of the DigitMCMetaData
  float mPedestal = 0.f;   ///< Pedestal value of the DigitMCMetaData
  float mNoise = 0.f;      ///< Noise value of the DigitMCMetaData

  ClassDefNV(DigitMCMetaData, 1);
};

inline DigitMCMetaData::DigitMCMetaData(float ADC, float commonMode, float pedestal, float noise)
  : mADC(ADC),
    mCommonMode(commonMode),
    mPedestal(pedestal),
    mNoise(noise)
{
}

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_DigitMCMetaData_H_
