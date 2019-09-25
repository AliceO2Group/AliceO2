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
#include "ZDCSimulation/MCLabel.h"
#include "SimulationDataFormat/MCTruthContainer.h"
#include "CommonDataFormat/InteractionRecord.h"
#include <vector>
#include <deque>

namespace o2
{
namespace zdc
{

class Digitizer
{
 public:
  struct ChannelBCDataF { // float (accumulable) version of the ChannelBCData
    std::array<float, NTimeBinsPerBC> data = {0.f};
  };
  struct ChannelDataF { //  float (accumulable) version of the ChannelData
    std::array<ChannelBCDataF, NBCReadOut> data = {};
  };
  struct BCCache {
    std::array<ChannelBCDataF, NChannels> bcdata;
    std::vector<o2::zdc::MCLabel> labels;
    o2::InteractionRecord intRecord;
  };

  // set event time
  void setEventID(int eventID) { mEventID = eventID; }
  void setSrcID(int sID) { mSrcID = sID; }
  void setInteractionRecord(const o2::InteractionTimeRecord& ir) { mIR = ir; }

  void process(const std::vector<o2::zdc::Hit>& hits,
               std::vector<o2::zdc::Digit>& digits,
               o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels);

  void flush(std::vector<o2::zdc::Digit>& digits, o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels);

 private:
  void phe2Sample(int nphe, double timeInSample, ChannelBCDataF& sample) const;
  int createDigit(std::vector<o2::zdc::Digit>& digits, o2::dataformats::MCTruthContainer<o2::zdc::MCLabel>& labels, int cachedID);
  BCCache& getCreateBCCache(const o2::InteractionRecord& ir);

  bool NeedToTrigger(const BCCache& bc) const; // TODO

  int mEventID = 0;
  int mSrcID = 0;
  o2::InteractionTimeRecord mIR;

  std::deque<BCCache> mCache; // cached BCs data

  ClassDefNV(Digitizer, 1);
};
} // namespace zdc
} // namespace o2

#endif /* DETECTORS_ZDC_SIMULATION_INCLUDE_ZDCSIMULATION_ZDCDIGITIZER_H_ */
