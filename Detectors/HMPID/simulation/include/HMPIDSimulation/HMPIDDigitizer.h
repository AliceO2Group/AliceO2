// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_
#define DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_

#include "HMPIDBase/Digit.h"
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
  void setEventTime(double timeNS) { mTime = timeNS; }
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

 private:
  void zeroSuppress(std::vector<o2::hmpid::Digit> const& digits, std::vector<o2::hmpid::Digit>& newdigits,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel>* newlabels);

  float getThreshold(o2::hmpid::Digit const&) const; // gives back threshold to apply for a certain digit
                                                     // (using noise and other tables for pad)

  double mTime = 0.;
  int mEventID = 0;
  int mSrcID = 0;

  // internal buffers for digits
  std::vector<o2::hmpid::Digit> mSummable;
  std::vector<o2::hmpid::Digit> mFinal;

  std::vector<o2::hmpid::Digit> mDigits; // internal store for digits

  //static constexpr int HMPID_NUMBEROFPADS = 161280;
  //std::array<short, HMPID_NUMBEROFPADS> mIndexForPad = { -1 }; //! mapping of pad to digit index

  std::map<int, short> mIndexForPad; //! logarithmic mapping of pad to digit index

  std::vector<int> mInvolvedPads; //! list of pads where digits created

  // other stuff needed for digitization
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mTmpLabelContainer;                   // temp label container as workspace
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mRegisteredLabelContainer = nullptr; // label container to be filled

  ClassDefNV(HMPIDDigitizer, 1);
};
} // namespace hmpid
} // namespace o2

#endif /* DETECTORS_HMPID_SIMULATION_INCLUDE_HMPIDSIMULATION_HMPIDDIGITIZER_H_ */
