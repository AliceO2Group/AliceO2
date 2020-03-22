// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FV0_DIGITIZATION_CONSTANT
#define ALICEO2_FV0_DIGITIZATION_CONSTANT

namespace o2::fv0
{
struct DigitizationConstant {
  static constexpr int NCELLSA = 40;                        // number of scintillator cells
  static constexpr int NCHANNELS = 48;                      // number of readout channels
  static constexpr float INV_CHARGE_PER_ADC = 1. / 0.6e-12; // charge
  static constexpr float N_PHOTONS_PER_MEV = 10400;         // average #photons generated per 1 MeV of deposited energy

  //TODO: optimize random ring sizes to balance between sim quality and execution time
  static constexpr int PHE_RANDOM_RING_SIZE = 1e5;  // size of random ring to be used inside photoelectron loop
  static constexpr int HIT_RANDOM_RING_SIZE = 1e4;  // size of random ring to be used inside hit loop
  static constexpr int NUM_PMT_RESPONSE_TABLES = 9; // number of PMT response tables
};
} // namespace o2::fv0
#endif
