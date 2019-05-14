// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_ZDCDIGITIZER_H_
#define DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_ZDCDIGITIZER_H_

#include "ZDCSimulation/Digit.h"
#include "ZDCSimulation/Hit.h" // for the hit
#include "SimulationDataFormat/MCTruthContainer.h"
#include "SimulationDataFormat/MCCompLabel.h"
#include <vector>

namespace o2
{
namespace zdc
{

class Digitizer
{
 public:
  // set event time
  void setEventTime(double timeNS) { mEventTime = timeNS; }
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }

  // user can pass a label container to be filled ... this activates the label mechanism
  // the passed label container can be readout after call to process
  void setLabelContainer(o2::dataformats::MCTruthContainer<o2::MCCompLabel>* labels)
  {
    mRegisteredLabelContainer = labels;
  }

  short fromPhotoelectronsToACD(int nphotoelectrons) const { return nphotoelectrons * (-0.0333333); } // TODO: implement real conversion method

  int toChannel(char detID, char secID) const { return detID * 5 + secID; }

  // this will process hits and fill the digit vector with digits which are finalized
  void process(std::vector<o2::zdc::Hit> const&, std::vector<o2::zdc::Digit>& digit);

  void zeroSuppress(std::vector<o2::zdc::Digit> const& digits, std::vector<o2::zdc::Digit>& newdigits,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel> const& labels,
                    o2::dataformats::MCTruthContainer<o2::MCCompLabel>* newlabels);

  // flush accumulated digits into the given container
  void flush(std::vector<o2::zdc::Digit>& digit);
  // reset internal data structures
  void reset();

 private:
  double mEventTime = 0;
  int mEventID = 0;
  int mSrcID = 0;

  std::vector<o2::zdc::Digit> mDigits; // internal store for digits

  // other stuff needed for digitization
  o2::dataformats::MCTruthContainer<o2::MCCompLabel> mTmpLabelContainer;                   // temp label container as workspace
  o2::dataformats::MCTruthContainer<o2::MCCompLabel>* mRegisteredLabelContainer = nullptr; // label container to be filled

  float getThreshold(o2::zdc::Digit const& digiti) const;

  ClassDefNV(Digitizer, 1);
};
} // namespace zdc
} // namespace o2

#endif /* DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_ZDCDIGITIZER_H_ */
