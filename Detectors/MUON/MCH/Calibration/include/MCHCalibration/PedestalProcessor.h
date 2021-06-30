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

/** @file PedestalsProcessor.h
 * C++ helper class to compute the mean and RMS of the pedestals
 * @author  Andrea Ferrero
 */

#ifndef ALICEO2_MCH_CALIBRATION_PEDESTAL_PROCESSOR_H_
#define ALICEO2_MCH_CALIBRATION_PEDESTAL_PROCESSOR_H_

#include <array>
#include <unordered_map>
#include <gsl/span>

#include "Rtypes.h"
#include "MCHCalibration/PedestalDigit.h"

namespace o2
{
namespace mch
{
namespace calibration
{

// \class PedestalProcessor
/// \brief helper class to compute the mean and RMS of the pedestals
class PedestalProcessor
{
 public:
  struct PedestalRecord {
    int mEntries{0};
    double mPedestal{0};
    double mVariance{0};

    double getRms();
  };

  using PedestalMatrix = std::array<std::array<PedestalRecord, 64>, 40>;
  using PedestalsMap = std::unordered_map<int, PedestalMatrix>;

  PedestalProcessor();

  ~PedestalProcessor() = default;

  void process(gsl::span<const PedestalDigit> digits);
  void reset();

  const PedestalsMap& getPedestals() const { return mPedestals; }

  uint32_t getMaxDsId() { return 39; }
  uint32_t getMaxChannel() { return 63; }

 private:
  PedestalsMap mPedestals;
}; //class PedestalProcessor

} //namespace calibration
} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_CALIBRATION_PEDESTAL_PROCESSOR_H_
