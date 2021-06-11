// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_CPV_DIGITIZER_H
#define ALICEO2_CPV_DIGITIZER_H

#include "DataFormatsCPV/Digit.h"
#include "CPVBase/Geometry.h"
#include "CPVCalib/CalibParams.h"
#include "CPVCalib/Pedestals.h"
#include "CPVCalib/BadChannelMap.h"
#include "CPVBase/Hit.h"
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

  /// Steer conversion of hits to digits
  void processHits(const std::vector<Hit>* mHits, const std::vector<Digit>& digitsBg,
                   std::vector<Digit>& digitsOut, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mLabels,
                   int source, int entry, double dt);

 protected:
  float simulatePedestalNoise(int absId);

 private:
  static constexpr short NCHANNELS = 23040;  //128*60*3:  toatl number of CPV channels
  std::unique_ptr<CalibParams> mCalibParams; /// Calibration coefficients
  std::unique_ptr<Pedestals> mPedestals;     /// Pedestals
  std::unique_ptr<BadChannelMap> mBadMap;    /// Bad channel map
  std::array<Digit, NCHANNELS> mArrayD;      ///array of digits (for inner use)
  std::array<float, NCHANNELS> mDigitThresholds;
  ClassDefOverride(Digitizer, 3);
};
} // namespace cpv
} // namespace o2

#endif /* ALICEO2_CPV_DIGITIZER_H */
