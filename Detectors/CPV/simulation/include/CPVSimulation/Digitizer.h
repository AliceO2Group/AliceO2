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

#ifndef ALICEO2_CPV_DIGITIZER_H
#define ALICEO2_CPV_DIGITIZER_H

#include "CPVBase/Geometry.h"
#include "DataFormatsCPV/Hit.h"
#include "DataFormatsCPV/Digit.h"
#include "DataFormatsCPV/CalibParams.h"
#include "DataFormatsCPV/Pedestals.h"
#include "DataFormatsCPV/BadChannelMap.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace cpv
{
class Digitizer : public TObject
{
 public:
  Digitizer() = default;
  ~Digitizer() override = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void init();
  void finish();
  void setPedestals(const Pedestals* peds) { mPedestals = peds; }
  void setBadChannelMap(const BadChannelMap* bcm) { mBadMap = bcm; }
  void setGains(const CalibParams* gains) { mGains = gains; }

  /// Steer conversion of hits to digits
  void processHits(const std::vector<Hit>* mHits, const std::vector<Digit>& digitsBg,
                   std::vector<Digit>& digitsOut, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mLabels,
                   int source, int entry, double dt);

 protected:
  float simulatePedestalNoise(int absId);

 private:
  static constexpr short NCHANNELS = 23040;      // 128*60*3:  toatl number of CPV channels
  const CalibParams* mGains = nullptr;           /// Calibration coefficients
  const Pedestals* mPedestals = nullptr;         /// Pedestals
  const BadChannelMap* mBadMap = nullptr;        /// Bad channel map
  std::array<Digit, NCHANNELS> mArrayD;          /// array of digits (for inner use)
  std::array<float, NCHANNELS> mDigitThresholds; /// array of readout thresholds (for inner use)
  ClassDefOverride(Digitizer, 3);
};
} // namespace cpv
} // namespace o2

#endif /* ALICEO2_CPV_DIGITIZER_H */
