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

struct SignalArray {
  std::array<float, kTimeBins> signals{};
  size_t labelIndex{0};
};

using DigitContainer = std::vector<Digit>;
using SignalContainer = std::unordered_map<int, SignalArray>;

class Digitizer
{
 public:
  Digitizer();
  ~Digitizer() = default;
  void process(std::vector<HitType> const&, DigitContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setEventID(int entryID) { mEventID = entryID; }
  void setSrcID(int sourceID) { mSrcID = sourceID; }
  void setCalibrations(Calibrations* calibrations) { mCalib = calibrations; }

  int getEventTime() const { return mTime; }
  int getEventID() const { return mEventID; }
  int getSrcID() const { return mSrcID; }

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
  bool convertHits(const int, const std::vector<HitType>&, SignalContainer&, o2::dataformats::MCTruthContainer<MCLabel>&, int thread = 0); // True if hit-to-signal conversion is successful
  bool convertSignalsToADC(const int, SignalContainer&, DigitContainer&, int thread = 0);                                                  // True if signal-to-ADC conversion is successful

  bool diffusion(float, float, float, float, float, float, double&, double&, double&, int thread = 0); // True if diffusion is applied successfully

  // Helpers for signal handling
  static constexpr int KEY_MIN = 0;
  static constexpr int KEY_MAX = 2211727;
  int calculateKey(const int det, const int row, const int col) { return ((det << 12) | (row << 8) | col); }
  int getDetectorFromKey(const int key) { return (key >> 12) & 0xFFF; }
  int getRowFromKey(const int key) { return (key >> 8) & 0xF; }
  int getColFromKey(const int key) { return key & 0xFF; }
};
} // namespace trd
} // namespace o2
#endif
