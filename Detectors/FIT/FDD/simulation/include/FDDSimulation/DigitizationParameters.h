// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FDD_DIGITIZATION_PARAMETERS
#define ALICEO2_FDD_DIGITIZATION_PARAMETERS

namespace o2::fdd
{
struct DigitizationParameters {
  static constexpr UShort_t mNchannels = 16;
  static constexpr UShort_t mNtriggers = 5;
  static constexpr Float_t mIntTimeRes = 0.4;
  static constexpr Float_t mPhotoCathodeEfficiency = 0.18;
  static constexpr Float_t mLightYield = 0.01;
  static constexpr Float_t mPmGain = 1e6;
  static constexpr Float_t mChargePerADC = 0.6e-12;
  static constexpr Float_t mPMTransitTime = 6.0;   // PM response time (corresponds to 1.9 ns rise time)
  static constexpr Float_t mPMTransparency = 0.25; // Transparency of the first dynode of the PM
  static constexpr Float_t mPMNbOfSecElec = 6.0;   // Number of secondary electrons emitted from first dynode (per ph.e.)
  static constexpr Float_t mShapeAlpha = -0.445;
  static constexpr Float_t mShapeN = 2.65;
  static constexpr Float_t mShapeSigma = 3.25;
  //static constexpr Float_t mPedestal = 0;
  static constexpr Float_t mTimeShiftCFD = 1.42;
  static constexpr int mPheRRSize = 1e5;     // size of random ring to be used inside photoelectron loop
  static constexpr int mHitRRSize = 1e4;     // size of random ring to be used inside hit loop
  static constexpr int mNResponseTables = 9; // number of PMT response tables
};
} // namespace o2::fdd
#endif
