// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_PHOS_DIGITIZER_H
#define ALICEO2_PHOS_DIGITIZER_H

#include "DataFormatsPHOS/Digit.h"
#include "DataFormatsPHOS/CalibParams.h"
#include "DataFormatsPHOS/TriggerMap.h"
#include "DataFormatsPHOS/MCLabel.h"
#include "PHOSBase/Geometry.h"
#include "PHOSBase/Hit.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace phos
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
                   std::vector<Digit>& digitsOut, o2::dataformats::MCTruthContainer<MCLabel>& mLabels,
                   int source, int entry, double dt);
  void processMC(bool mc) { mProcessMC = mc; }

 protected:
  float nonLinearity(float e);
  float uncalibrate(float e, int absId);
  float uncalibrateT(float t, int absId);
  float timeResolution(float time, float e);
  float simulateNoiseEnergy(int absId);
  float simulateNoiseTime();

 private:
  static constexpr short NCHANNELS = 12544; ///< Number of channels starting from 56*64*(4-0.5)
  static constexpr short OFFSET = 1793;     ///< Non-existing channels 56*64*0.5+1
  bool mProcessMC = true;
  bool mTrig2x2 = true;                      ///< simulate 2x2 PHOS trigger
  bool mTrig4x4 = false;                     ///< simulate 4x4 PHOS trigger
  std::unique_ptr<CalibParams> mCalibParams; /// Calibration coefficients
  std::unique_ptr<TriggerMap> mTrigUtils;    /// trigger bad map and turn-on curves
  std::array<Digit, NCHANNELS> mArrayD;

  ClassDefOverride(Digitizer, 4);
};
} // namespace phos
} // namespace o2

#endif /* ALICEO2_PHOS_DIGITIZER_H */
