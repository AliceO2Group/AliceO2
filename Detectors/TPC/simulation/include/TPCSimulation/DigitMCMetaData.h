// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
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
#include <TObject.h>

namespace o2 {
namespace TPC {


/// \class DigitMCMetaData
/// This is the definition of the Monte Carlo Digit object, which is the final entity after Digitization
/// Its coordinates are defined by the CRU, the time bin, the pad row and the pad.
/// It holds the ADC value of a given pad on the pad plane.
/// Additional information attached to it are the MC label of the contributing tracks

class DigitMCMetaData : public TObject {
  public:

    /// Default constructor
    DigitMCMetaData();

    /// Constructor, initializing values for position, charge, time and common mode
    /// \param ADC Raw ADC value of the corresponding DigitMC (before saturation)
    /// \param commonMode Common mode signal on that ROC in the time bin of the DigitMC
    /// \param pedestal Raw pedestal value of the corresponding DigitMC (before saturation)
    /// \param noise Raw noise value of the corresponding DigitMC (before saturation)
    DigitMCMetaData(float ADC, float commonMode, float pedestal, float noise);

    /// Destructor
    virtual ~DigitMCMetaData() = default;

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
    float         mADC;             ///< Raw ADC value of the DigitMCMetaData
    float         mCommonMode;      ///< Common mode value of the DigitMCMetaData
    float         mPedestal;        ///< Pedestal value of the DigitMCMetaData
    float         mNoise;           ///< Noise value of the DigitMCMetaData

  ClassDef(DigitMCMetaData, 1);
};

inline
DigitMCMetaData::DigitMCMetaData()
  : mADC(0.f),
    mCommonMode(0.f),
    mPedestal(0.f),
    mNoise(0.f)
  {}

inline
DigitMCMetaData::DigitMCMetaData(float ADC, float commonMode, float pedestal, float noise)
  : mADC(ADC),
    mCommonMode(commonMode),
    mPedestal(pedestal),
    mNoise(noise)
{}

}
}

#endif // ALICEO2_TPC_DigitMCMetaData_H_
