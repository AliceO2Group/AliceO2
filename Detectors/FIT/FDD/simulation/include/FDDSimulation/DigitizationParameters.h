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
  const UShort_t mNchannels = 16;
  const UShort_t mNtriggers = 5;
  const Float_t mIntTimeRes = 0.4;
  const Float_t mPhotoCathodeEfficiency = 0.18;
  const Float_t mLightYield = 0.01;
  const Float_t mPmGain = 1e6;
  const Float_t mChargePerADC = 0.6e-12;
  const Float_t mPMTransitTime = 6.0;   // PM response time (corresponds to 1.9 ns rise time)
  const Float_t mPMTransparency = 0.25; // Transparency of the first dynode of the PM
  const Float_t mPMNbOfSecElec = 6.0;   // Number of secondary electrons emitted from first dynode (per ph.e.)
};
} // namespace o2::fdd
#endif
