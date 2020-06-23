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
  std::array<float, kTimeBins> signals{}; // signals
  double firstTBtime;                     // first TB time
  std::unordered_map<int, int> trackIds;  // tracks Ids associated to the signal
  std::vector<MCLabel> labels;            // labels associated to the signal
};

using DigitContainer = std::vector<Digit>;
using SignalContainer = std::unordered_map<int, SignalArray>;

class Digitizer
{
 public:
  Digitizer() = default;
  ~Digitizer() = default;
  void init(); // setup everything

  void process(std::vector<HitType> const&, DigitContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
  void flush(DigitContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
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
  double mLastTime = 1.0e10; // starts in the future
  int mEventID = 0;
  int mSrcID = 0;

  // Digitization parameters
  static constexpr float AmWidth = TRDGeometry::amThick(); // Width of the amplification region
  static constexpr float DrWidth = TRDGeometry::drThick(); // Width of the drift retion
  static constexpr float DrMin = -0.5 * AmWidth;           // Drift + Amplification region
  static constexpr float DrMax = DrWidth + 0.5 * AmWidth;  // Drift + Amplification region
  float mSamplingRate = 0;                                 // The sampling rate
  float mElAttachProp = 0;                                 // Propability for electron attachment (for 1m)
  int mNpad = 0;                                           // Number of pads included in the pad response
  int mTimeBinTRFend = 0;                                  // time bin TRF ends
  int mMaxTimeBins = 30;                                   // Maximum number of time bins for processing signals, usually set at 30 tb = 3 microseconds
  int mMaxTimeBinsTRAP = 30;                               // Maximum number of time bins for processing adcs; should be read from the CCDB or the TRAP config

  // Digitization containers
  std::vector<HitType> mHitContainer;                       // the container of hits in a given detector
  std::vector<MCLabel> mMergedLabels;                       // temporary label container
  std::array<SignalContainer, kNdet> mSignalsMapCollection; // container for caching signals over a timeframe
  std::array<DigitContainer, kNdet> mDigitsCollection;      // container for caching digits for paralellization

  void getHitContainerPerDetector(const std::vector<HitType>&, std::array<std::vector<HitType>, kNdet>&);
  void clearCollections();
  void setSimulationParameters();

  // Digitization chain methods
  bool convertHits(const int, const std::vector<HitType>&, SignalContainer&, int thread = 0); // True if hit-to-signal conversion is successful
  bool convertSignalsToADC(const int, SignalContainer&, DigitContainer&, int thread = 0);     // True if signal-to-ADC conversion is successful
  void addLabel(const o2::trd::HitType& hit, std::vector<o2::trd::MCLabel>&, std::unordered_map<int, int>&);
  bool diffusion(float, float, float, float, float, float, double&, double&, double&, int thread = 0); // True if diffusion is applied successfully

  // Helpers for signal handling
  static constexpr int KEY_MIN = 0;
  static constexpr int KEY_MAX = 2211727;
  int calculateKey(const int det, const int row, const int col)
  {
    int key = ((det << 12) | (row << 8) | col);
    assert(!(key < KEY_MIN || key > KEY_MAX));
    return key;
  }
  int getDetectorFromKey(const int key) { return (key >> 12) & 0xFFF; }
  int getRowFromKey(const int key) { return (key >> 8) & 0xF; }
  int getColFromKey(const int key) { return key & 0xFF; }
};
} // namespace trd
} // namespace o2
#endif
