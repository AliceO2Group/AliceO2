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

#ifndef ALICEO2_FDD_DIGITIZATION_PARAMETERS
#define ALICEO2_FDD_DIGITIZATION_PARAMETERS

namespace o2::fdd
{
struct DigitizationParameters {
  static constexpr float LightYield = 0.01;
  static constexpr float PmGain = 1e6;
  static constexpr float ShapeAlpha = -0.445;
  static constexpr float ShapeN = 2.65;
  static constexpr float ShapeSigma = 3.25;
  // static constexpr float Pedestal = 0;
  static constexpr float TimeShiftCFD = 1.42;
  static constexpr float TimeDelayFDA = 30.0;
  static constexpr float TimeDelayFDC = 30.0;
  static constexpr int PheRRSize = 1e5;           // size of random ring to be used inside photoelectron loop
  static constexpr int HitRRSize = 1e4;           // size of random ring to be used inside hit loop
  static constexpr int NResponseTables = 9;       // number of PMT response tables
  static constexpr uint8_t defaultFEEbits = 0x48; // (01001000) only 2 flags are set by default in simulation:kIsCFDinADCgate and kIsEventInTVDC
};
} // namespace o2::fdd
#endif
