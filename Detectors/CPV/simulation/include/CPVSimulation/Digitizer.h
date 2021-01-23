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
  void addNoisyChannels(int start, int end, std::vector<Digit>& digitsOut);
  float uncalibrate(float e, int absId);
  float simulateNoise();

 private:
  std::unique_ptr<CalibParams> mCalibParams; /// Calibration coefficients

  ClassDefOverride(Digitizer, 2);
};
} // namespace cpv
} // namespace o2

#endif /* ALICEO2_CPV_DIGITIZER_H */
