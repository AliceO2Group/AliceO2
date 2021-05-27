// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file Digitizer.h
 * C++  MCH Digitizer.
 * @author Michael Winn, Laurent Aphecetche
 */

#ifndef O2_MCH_SIMULATION_MCHDIGITIZER_H_
#define O2_MCH_SIMULATION_MCHDIGITIZER_H_

#include "DataFormatsMCH/Digit.h"
#include "MCHSimulation/Hit.h"

#include "SimulationDataFormat/MCCompLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "DataFormatsMCH/ROFRecord.h"

namespace o2
{
namespace mch
{

class Digitizer
{
 public:
  Digitizer(int mode = 0);

  ~Digitizer() = default;

  void init();

  //process hits: fill digit vector with digits
  void process(const std::vector<Hit> hits, std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer);
  void provideMC(o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer);
  void mergeDigits(std::vector<Digit>& rofdigits, std::vector<o2::MCCompLabel>& rofLabels, std::vector<int>& indexhelper);
  void generateNoiseDigits();
  //external pile-up adding up
  void mergeDigits(std::vector<Digit>& digits, o2::dataformats::MCTruthContainer<o2::MCCompLabel>& mcContainer, std::vector<ROFRecord>& rofs);

  void fillOutputContainer(std::vector<Digit>& digits);

  void setEventTime(double timeNS) { mEventTime = timeNS; }

  void setContinuous(bool val) { mContinuous = val; }
  bool isContinuous() const { return mContinuous; }

  void setSrcID(int v);
  int getSrcID() const { return mSrcID; }

  void setEventID(int v);
  int getEventID() const { return mEventID; }

  void setNoise(bool val) { mNoise = val; }
  bool isNoise() const { return mNoise; }

  //for debugging
  std::vector<Digit> getDigits() { return mDigits; }
  std::vector<o2::MCCompLabel> getTrackLabels() { return mTrackLabels; }

 private:
  int mEventTime;
  int mEventID = 0;
  int mSrcID = 0;

  bool mContinuous = false;
  bool mNoise = true;

  //time difference allowed for pileup (in ns (assuming that event time is in ns))
  int mDeltat = 4;

  //number of detector elements
  const static int mNdE = 156;

  //noise above threshold probability within read-out window
  float mProbNoise = 1e-5;
  //sum_i 1/padcount_i where i is the detelemID
  float mInvPadSum = 0.0450832;
  float mNormProbNoise = mProbNoise / mInvPadSum;

  // digit per pad
  std::vector<Digit> mDigits;

  //MCLabel container (transient)
  std::vector<o2::MCCompLabel> mTrackLabels;
  //MCLabel container (output)
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mMCTruthOutputContainer;

  int processHit(const Hit& hit, int detID, int eventTime);
};

} // namespace mch
} // namespace o2
#endif
