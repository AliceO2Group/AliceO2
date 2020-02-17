// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FV0_DIGITIZATION_PARAMETERS
#define ALICEO2_FV0_DIGITIZATION_PARAMETERS

namespace o2::fv0
{
struct DigitizationParameters {
  static constexpr int NCELLSA = 40;                        // number of scintillator cells
  static constexpr int NCHANNELS = 48;                      // number of readout channels
  static constexpr float INV_CHARGE_PER_ADC = 1. / 0.6e-12; // charge
  static constexpr float N_PHOTONS_PER_MEV = 10400;         // average #photons generated per 1 MeV of deposited energy

  //TODO: optimize random ring sizes to balance between sim quality and execution time
  static constexpr int PHE_RANDOM_RING_SIZE = 1e5;  // size of random ring to be used inside photoelectron loop
  static constexpr int HIT_RANDOM_RING_SIZE = 1e4;  // size of random ring to be used inside hit loop
  static constexpr int NUM_PMT_RESPONSE_TABLES = 9; // number of PMT response tables

  //TODO: move all params below to configurable params
  static constexpr float mIntrinsicTimeRes = 0.91;       // time resolution
  static constexpr float mPhotoCathodeEfficiency = 0.23; // quantum efficiency = nOfPhotoE_emitted_by_photocathode / nIncidentPhotons
  static constexpr float mLightYield = 0.1;              // light collection efficiency to be tuned using collision data
  static constexpr float mPmtGain = 5e4;                 // value for PMT R5924-70 at default FV0 gain
  static constexpr float mPmtTransitTime = 9.5;          // PMT response time (corresponds to 1.9 ns rise time)
  static constexpr float mPmtTransparency = 0.25;        // Transparency of the first dynode of the PMT
  static constexpr float mPmtNbOfSecElec = 9.0;          // Number of secondary electrons emitted from first dynode (per ph.e.)
  static constexpr float mShapeAlpha = -0.445;
  static constexpr float mShapeN = 2.65;
  static constexpr float mShapeSigma = 3.25;
  //const Float_t mPedestal = 0;
  static constexpr float mTimeShiftCfd = 1.42;     // TODO: adjust after PM design for FV0 is fixed
  static constexpr int photoelMin = 0;             // Integration lower limit
  static constexpr int photoelMax = 30;            // Integration upper limit
  static constexpr float singleMipThreshold = 3.0; // in [MeV] of deposited energy
  // Optimization-related, derived constants
  static constexpr float mOneOverPmtTransitTime2 = 1.0 / (mPmtTransitTime * mPmtTransitTime);
};
} // namespace o2::fv0
#endif
