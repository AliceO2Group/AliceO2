// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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

static const size_t MCH_NUMBER_OF_SOLAR = 100 * 8;

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

  double getPedestal(uint32_t solarId, uint32_t dsId, uint32_t channel) const;
  double getRms(uint32_t solarId, uint32_t dsId, uint32_t channel) const;

  uint32_t getMaxSolarId() { return (MCH_NUMBER_OF_SOLAR - 1); }
  uint32_t getMaxDsId() { return 39; }
  uint32_t getMaxChannel() { return 63; }

 private:
  //uint64_t mNhits[MCH_NUMBER_OF_SOLAR][40][64];
  //double mPedestal[MCH_NUMBER_OF_SOLAR][40][64];
  //double mNoise[MCH_NUMBER_OF_SOLAR][40][64];

  PedestalsMap mPedestals;
}; //class PedestalProcessor

} //namespace calibration
} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_CALIBRATION_PEDESTAL_PROCESSOR_H_
