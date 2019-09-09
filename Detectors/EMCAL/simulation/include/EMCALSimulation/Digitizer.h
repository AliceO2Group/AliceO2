// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_EMCAL_DIGITIZER_H
#define ALICEO2_EMCAL_DIGITIZER_H

#include <memory>
#include <unordered_map>
#include <vector>
#include <list>

#include "Rtypes.h"  // for Digitizer::Class, Double_t, ClassDef, etc
#include "TObject.h" // for TObject
#include "TRandom3.h"

#include "DataFormatsEMCAL/Digit.h"
#include "EMCALBase/Geometry.h"
#include "EMCALBase/GeometryBase.h"
#include "EMCALBase/Hit.h"
#include "EMCALSimulation/SimParam.h"
#include "EMCALSimulation/LabeledDigit.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"

namespace o2
{
namespace emcal
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
  void process(const std::vector<Hit>& hits, std::vector<Digit>& digits);

  void setEventTime(double t);
  double getEventTime() const { return mEventTime; }

  void setContinuous(bool v) { mContinuous = v; }
  bool isContinuous() const { return mContinuous; }

  void fillOutputContainer(std::vector<Digit>& digits);

  void setSmearEnergy(bool v) { mSmearEnergy = v; }
  bool doSmearEnergy() const { return mSmearEnergy; }
  void smearEnergy(LabeledDigit& digit);

  void setRemoveDigitsBelowThreshold(bool v) { mRemoveDigitsBelowThreshold = v; }
  bool doRemoveDigitsBelowThreshold() const { return mRemoveDigitsBelowThreshold; }

  void setSimulateNoiseDigits(bool v) { mSimulateNoiseDigits = v; }
  bool doSimulateNoiseDigits() const { return mSimulateNoiseDigits; }
  void addNoiseDigits();

  void setCoeffToNanoSecond(double cf) { mCoeffToNanoSecond = cf; }
  double getCoeffToNanoSecond() const { return mCoeffToNanoSecond; }

  void setCurrSrcID(int v);
  int getCurrSrcID() const { return mCurrSrcID; }

  void setCurrEvID(int v);
  int getCurrEvID() const { return mCurrEvID; }

  void setGeometry(const o2::emcal::Geometry* gm) { mGeometry = gm; }

  void hitToDigits(const Hit& hit);

  static double rawResponseFunction(double* x, double* par);
  /// raw pointers used here to allow interface with TF1

 private:
  const Geometry* mGeometry = nullptr;     ///< EMCAL geometry
  double mEventTime = 0;                   ///< global event time
  double mCoeffToNanoSecond = 1.0;         ///< coefficient to convert event time (Fair) to ns
  bool mContinuous = false;                ///< flag for continuous simulation
  UInt_t mROFrameMin = 0;                  ///< lowest RO frame of current digits
  UInt_t mROFrameMax = 0;                  ///< highest RO frame of current digits
  int mCurrSrcID = 0;                      ///< current MC source from the manager
  int mCurrEvID = 0;                       ///< current event ID from the manager
  bool mSmearEnergy = true;                ///< do time and energy smearing
  bool mSimulateTimeResponse = true;       ///< simulate time response
  bool mRemoveDigitsBelowThreshold = true; ///< remove digits below threshold
  bool mSimulateNoiseDigits = true;        ///< simulate noise digits
  const SimParam* mSimParam = nullptr;     ///< SimParam object

  std::vector<Digit> mTempDigitVector;                          ///< temporary digit storage
  std::unordered_map<Int_t, std::list<LabeledDigit>> mDigits;   ///< used to sort digits and labels by tower
  o2::dataformats::MCTruthContainer<MCLabel> mMCTruthContainer; ///< contains MC truth information

  TRandom3* mRandomGenerator = nullptr; // random number generator
  Int_t mTimeBinOffset = 0;             // offset of first time bin
  std::vector<double> mSignalFractionInTimeBins; // fraction of signal for each time bin

  ClassDefOverride(Digitizer, 1);
};
} // namespace emcal
} // namespace o2

#endif /* ALICEO2_EMCAL_DIGITIZER_H */
