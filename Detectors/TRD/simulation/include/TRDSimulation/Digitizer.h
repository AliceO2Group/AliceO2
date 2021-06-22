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

#include "TRDBase/Calibrations.h"
#include "TRDBase/CommonParam.h"
#include "TRDBase/DiffAndTimeStructEstimator.h"
#include "TRDSimulation/PileupTool.h"

#include "DataFormatsTRD/Digit.h"
#include "DataFormatsTRD/SignalArray.h"
#include "DataFormatsTRD/Constants.h"

#include "MathUtils/RandomRing.h"
#include "SimulationDataFormat/MCTruthContainer.h"

#include <array>
#include <deque>
#include <unordered_map>
#include <vector>

namespace o2
{
namespace trd
{

class Geometry;
class SimParam;
class PadPlane;
class TRDArraySignal;
class PadResponse;

using DigitContainer = std::vector<Digit>;
using SignalContainer = std::unordered_map<int, SignalArray>;
using MCLabel = o2::MCCompLabel;

class Digitizer
{
 public:
  Digitizer() = default;
  ~Digitizer() = default;
  void init(); // setup everything

  void process(std::vector<Hit> const&);
  void flush(DigitContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
  void dumpLabels(const SignalContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
  void pileup();
  void setEventTime(double timeNS) { mTime = timeNS; }
  void setTriggerTime(double t) { mCurrentTriggerTime = t; }
  void setEventID(int entryID) { mEventID = entryID; }
  void setSrcID(int sourceID) { mSrcID = sourceID; }
  void setCalibrations(Calibrations* calibrations) { mCalib = calibrations; }
  void setCreateSharedDigits(bool flag) { mCreateSharedDigits = flag; }
  int getEventTime() const { return mTime; }
  int getEventID() const { return mEventID; }
  int getSrcID() const { return mSrcID; }
  bool getCreateSharedDigits() const { return mCreateSharedDigits; }

 private:
  Geometry* mGeo = nullptr;               // access to Geometry
  PadResponse* mPRF = nullptr;            // access to PadResponse
  SimParam* mSimParam = nullptr;          // access to SimParam instance
  CommonParam* mCommonParam = nullptr;    // access to CommonParam instance
  Calibrations* mCalib = nullptr;         // access to Calibrations in CCDB
  PileupTool pileupTool;

  // number of digitizer threads
  int mNumThreads = 1;

  // we create one such service structure per thread
  std::vector<math_utils::RandomRing<>> mGausRandomRings; // pre-generated normal distributed random numbers
  std::vector<math_utils::RandomRing<>> mFlatRandomRings; // pre-generated flat distributed random numbers
  std::vector<math_utils::RandomRing<>> mLogRandomRings;  // pre-generated exp distributed random number
  std::vector<DiffusionAndTimeStructEstimator> mDriftEstimators;

  double mTime = 0.;               // time in nanoseconds of the hits currently being processed
  double mCurrentTriggerTime = 0.; // time in nanoseconds of the current trigger
  int mEventID = 0;                // event id
  int mSrcID = 0;                  // source id

  // Digitization parameters
  static constexpr float AmWidth = Geometry::amThick();    // Width of the amplification region
  static constexpr float DrWidth = Geometry::drThick();    // Width of the drift retion
  static constexpr float DrMin = -0.5 * AmWidth;           // Drift + Amplification region
  static constexpr float DrMax = DrWidth + 0.5 * AmWidth;  // Drift + Amplification region
  float mSamplingRate = 0;                                 // The sampling rate
  float mElAttachProp = 0;                                 // Propability for electron attachment (for 1m)
  int mNpad = 0;                                           // Number of pads included in the pad response
  int mTimeBinTRFend = 0;                                  // time bin TRF ends
  int mMaxTimeBins = 30;                                   // Maximum number of time bins for processing signals, usually set at 30 tb = 3 microseconds
  int mMaxTimeBinsTRAP = 30;                               // Maximum number of time bins for processing adcs; should be read from the CCDB or the TRAP config
  bool mCreateSharedDigits = true;                         // flag if copies should be created of digits from pads which are shared between MCMs

  // Digitization containers
  std::vector<Hit> mHitContainer;                                                // the container of hits in a given detector
  std::vector<MCLabel> mMergedLabels;                                            // temporary label container
  std::array<SignalContainer, constants::MAXCHAMBER> mSignalsMapCollection;      // container for caching signals over a timeframe
  std::deque<std::array<SignalContainer, constants::MAXCHAMBER>> mPileupSignals; // container for piled up signals

  void getHitContainerPerDetector(const std::vector<Hit>&, std::array<std::vector<Hit>, constants::MAXCHAMBER>&);
  void setSimulationParameters();

  // Digitization chain methods
  int triggerEventProcessing(DigitContainer&, o2::dataformats::MCTruthContainer<MCLabel>&);
  SignalContainer addSignalsFromPileup();
  void clearContainers();
  bool convertHits(const int, const std::vector<Hit>&, SignalContainer&, int thread = 0);              // True if hit-to-signal conversion is successful
  bool convertSignalsToADC(SignalContainer&, DigitContainer&, int thread = 0);                         // True if signal-to-ADC conversion is successful
  void addLabel(const int&, std::vector<MCLabel>&, std::unordered_set<int>&);                          // add a MC label, check if trackId is already registered
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
