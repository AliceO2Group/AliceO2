// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGITIZER_H_
#define ALICEO2_TRD_DIGITIZER_H_

#include "TRDSimulation/Detector.h"
#include "TRDBase/Digit.h"
#include "TRDBase/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include "TRDBase/TRDCommonParam.h"
#include "TRDBase/TRDDiffAndTimeStructEstimator.h"
#include "TRDBase/Calibrations.h"

#include "MathUtils/RandomRing.h"

namespace o2
{
namespace trd
{

class TRDGeometry;
class TRDSimParam;
class TRDPadPlane;
class TRDArraySignal;
class PadResponse;

class Digitizer
{
 public:
  Digitizer();
  ~Digitizer() = default;
  void process(std::vector<HitType> const&, DigitContainer_t&, o2::dataformats::MCTruthContainer<MCLabel>&);
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setEventID(int entryID) { mEventID = entryID; }
  void setSrcID(int sourceID) { mSrcID = sourceID; }
  void setCalibrations(Calibrations* calibrations) { mCalib = calibrations; }

 private:
  TRDGeometry* mGeo = nullptr;            // access to TRDGeometry
  PadResponse* mPRF = nullptr;            // access to PadResponse
  TRDSimParam* mSimParam = nullptr;       // access to TRDSimParam instance
  TRDCommonParam* mCommonParam = nullptr; // access to TRDCommonParam instance
  Calibrations* mCalib = nullptr;         // access to Calibrations in CCDB

  // number of digitizer threads
  int mNumThreads = 1;

  // we create one such service structure per thread
  std::vector<math_utils::RandomRing<>> mGausRandomRings; // pre-generated normal distributed random numbers
  std::vector<math_utils::RandomRing<>> mFlatRandomRings; // pre-generated flat distributed random numbers
  std::vector<math_utils::RandomRing<>> mLogRandomRings;  // pre-generated exp distributed random number

  std::vector<TRDDiffusionAndTimeStructEstimator> mDriftEstimators;

  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;

  bool mSDigits{false};               // true: convert signals to summable digits, false by defaults
  std::vector<HitType> mHitContainer; // the container of hits in a given detector

  void getHitContainerPerDetector(const std::vector<HitType>&, std::array<std::vector<HitType>, kNdet>&);
  // Digitization chaing methods
  bool convertHits(const int, const std::vector<HitType>&, SignalContainer_t&, o2::dataformats::MCTruthContainer<MCLabel>&, int thread = 0); // True if hit-to-signal conversion is successful
  bool convertSignalsToDigits(const int, SignalContainer_t&, int thread = 0);                                                                // True if signal-to-digit conversion is successful
  bool convertSignalsToSDigits(const int, SignalContainer_t&, int thread = 0);                                                               // True if signal-to-sdigit conversion is successful
  bool convertSignalsToADC(const int, SignalContainer_t&, int thread = 0);                                                                   // True if signal-to-ADC conversion is successful

  bool diffusion(float, float, float, float, float, float, double&, double&, double&, int thread = 0); // True if diffusion is applied successfully
};
} // namespace trd
} // namespace o2
#endif
