// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_FEEDIGITIZER_H
#define ALICEO2_EMCAL_FEEDIGITIZER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <list>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject
#include "TRandom3.h"

#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Hit.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALSimulation/LabeledDigit.h"
#include "EMCALSimulation/DigitsWriteoutBuffer.h"

#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace emcal
{

/// \class Digitizer
/// \brief EMCAL FEE digitizer
/// \ingroup EMCALsimulation
/// \author Anders Knospe, University of Houston
/// \author Hadi Hassan, ORNL
/// @TODO adapt it to digitize TRU digits
class Digitizer : public TObject
{
 public:
  Digitizer() = default;
  ~Digitizer() override = default;
  Digitizer(const Digitizer&) = delete;
  Digitizer& operator=(const Digitizer&) = delete;

  void init();
  void initCycle();
  void clear();
  void finish();

  /// Steer conversion of hits to digits
  void process(const std::vector<LabeledDigit>& labeledDigit);

  void setEventTime(double t);
  double getTriggerTime() const { return mTriggerTime; }
  double getEventTime() const { return mEventTime; }
  bool isLive(double t) const { return (t - mTriggerTime < mLiveTime); }
  bool isLive() const { return (mEventTime < mLiveTime); }

  bool isEmpty() const { return mEmpty; }

  void fillOutputContainer(std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::emcal::MCLabel>& labels);

  bool doSmearEnergy() const { return mSmearEnergy; }
  double smearEnergy(double energy);
  bool doSimulateTimeResponse() const { return mSimulateTimeResponse; }
  bool doRemoveDigitsBelowThreshold() const { return mRemoveDigitsBelowThreshold; }
  bool doSimulateNoiseDigits() const { return mSimulateNoiseDigits; }
  void addNoiseDigits(LabeledDigit&);

  void setCoeffToNanoSecond(double cf) { mCoeffToNanoSecond = cf; }
  double getCoeffToNanoSecond() const { return mCoeffToNanoSecond; }

  void sampleSDigit(const Digit& sdigit);

  static double rawResponseFunction(double* x, double* par);
  /// raw pointers used here to allow interface with TF1

 private:
  double mTriggerTime = -1e20;             ///< global trigger time
  double mEventTime = 0;                   ///< global event time
  short mEventTimeOffset = 0;              ///< event time difference from trigger time (in number of bins)
  short mPhase = 0;                        ///< event phase
  double mCoeffToNanoSecond = 1.0;         ///< coefficient to convert event time (Fair) to ns
  UInt_t mROFrameMin = 0;                  ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;                  ///< highest RO frame of current digits
  bool mSmearEnergy = true;                ///< do time and energy smearing
  bool mSimulateTimeResponse = true;       ///< simulate time response
  bool mRemoveDigitsBelowThreshold = true; ///< remove digits below threshold
  bool mSimulateNoiseDigits = true;        ///< simulate noise digits
  const SimParam* mSimParam = nullptr;     ///< SimParam object
  bool mEmpty = true;                      ///< Digitizer contains no digits/labels

  std::vector<Digit> mTempDigitVector;                        ///< temporary digit storage
  std::unordered_map<Int_t, std::list<LabeledDigit>> mDigits; ///< used to sort digits and labels by tower

  TRandom3* mRandomGenerator = nullptr;                  // random number generator
  std::vector<int> mTimeBinOffset;                       // offset of first time bin
  std::vector<std::vector<double>> mAmplitudeInTimeBins; // amplitude of signal for each time bin

  float mLiveTime = 1500;  // EMCal live time (ns)
  float mBusyTime = 35000; // EMCal busy time (ns)
  int mDelay = 7;          // number of (full) time bins corresponding to the signal time delay

  ClassDefOverride(Digitizer, 1);
};
} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_FEEDIGITIZER_H */
