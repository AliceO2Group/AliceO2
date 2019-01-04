// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Declaration of a transient MC label class for EMCal

#ifndef ALICEO2_EMCAL_MCLABEL_H_
#define ALICEO2_EMCAL_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace EMCAL
{
class MCLabel : public o2::MCCompLabel
{
 private:
  Double_t mEventTime;

 public:
  MCLabel() = default;
  MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, Int_t time) : o2::MCCompLabel(trackID, eventID, srcID), mEventTime(time) {}
  Double_t getEventTime() const { return mEventTime; }
  bool operator<(const MCLabel& other) const { return getEventTime() < other.getEventTime(); }
  bool operator>(const MCLabel& other) const { return getEventTime() > other.getEventTime(); }
  bool operator==(const MCLabel& other) const { return getEventTime() == other.getEventTime(); }
};
}
}

#endif
