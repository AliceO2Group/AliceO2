// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_FT0_DIGITIZATION_CONSTANTS
#define ALICEO2_FT0_DIGITIZATION_CONSTANTS

namespace o2::ft0
{
struct DigitizationConstants {
  static constexpr int NOISE_RANDOM_RING_SIZE = 1000 * 1000;
  static constexpr int SINC_TABLE_SIZE = 2048;
  static constexpr int SIGNAL_TABLE_SIZE = 4096;
  static constexpr double SIGNAL_CACHE_DT = 0.005;
};
} // namespace o2::ft0
#endif
