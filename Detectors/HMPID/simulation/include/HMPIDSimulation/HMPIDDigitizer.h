// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_
#define DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_

#include "DataFormatsHMP/Digit.h"
#include "DataFormatsHMP/Trigger.h"
#include "HMPIDSimulation/Detector.h" // for the hit
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>

namespace o2
{
namespace hmpid
{

class HMPIDDigitizer
{
 public:
  // set event time and return true if it is within the active hold/data-taking time
  // or false otherwise (in which case we don't need to do digitization)
  bool setEventTime(double timeNS)
  {
    if ((timeNS - mCurrentTriggerTime) > TRACKHOLDTIME) {
      return false;
    } else {
      return true;
    }
  }

  // set a trigger; returns true if accepted or false if busy
  // (assuming some extern decision on this time)
  bool setTriggerTime(double timeNS)
  {
    if (mReadoutCounter == -1) {
      // for the first trigger no busy check necessary
      mCurrentTriggerTime = timeNS;
      mReadoutCounter++;
      mBc = o2::InteractionRecord::ns2bc(mCurrentTriggerTime, mOrbit);
      return true;
    } else {
      if ((timeNS - mCurrentTriggerTime) > BUSYTIME) {
        mCurrentTriggerTime = timeNS;
        mBc = o2::InteractionRecord::ns2bc(mCurrentTriggerTime, mOrbit);
        mReadoutCounter++;
        return true;
      } else {
        return false;
      }
    }
  }

  uint32_t getOrbit() { return mOrbit; };
  uint16_t getBc() { return mBc; };

  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }

  // user can pass a label container to be filled ... this activates the label mechanism
  // the passed label container can be readout after call to process
  void setLabelContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels)
  {
    mRegisteredLabelContainer = labels;
  }

  // this will process hits and fill the digit vector with digits which are finalized
  void process(std::vector<o2::hmpid::HitType> const&, std::vector<o2::hmpid::Digit>& digit);

  // flush accumulated digits into the given container
  void flush(std::vector<o2::hmpid::Digit>& digit);
  // reset internal data structures
  void reset();

 private:
  void zeroSuppress(std::vector<o2::hmpid::Digit> const& digits, std::vector<o2::hmpid::Digit>& newdigits,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel>* newlabels);

  float getThreshold(o2::hmpid::Digit const&) const; // gives back threshold to apply for a certain digit
                                                     // (using noise and other tables for pad)

  double mCurrentTriggerTime = 0.;
  uint32_t mOrbit = 0;
  uint16_t mBc = 0;

  int mEventID = 0;
  int mSrcID = 0;

  std::vector<o2::hmpid::Digit> mDigits; // internal store for digits

  constexpr static double TRACKHOLDTIME = 1200; // defines the window for pile-up after a trigger received in nanoseconds
  constexpr static double BUSYTIME = 22000;     // the time for which no new trigger can be received in nanoseconds

  std::map<int, int> mIndexForPad; //! logarithmic mapping of pad to digit index

  std::vector<int> mInvolvedPads; //! list of pads where digits created

  int mReadoutCounter = -1;

  // other stuff needed for digitization
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mTmpLabelContainer;                   // temp label container as workspace
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mRegisteredLabelContainer = nullptr; // label container to be filled

  ClassDefNV(HMPIDDigitizer, 1);
};
} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_ */
