// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Declaration of a transient MC label class for FIT

#ifndef ALICEO2_FIT_MCLABEL_H_
#define ALICEO2_FIT_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace fit
{
class MCLabel : public o2::MCCompLabel
{
 private:
  Int_t mDetID = -1;

 public:
  MCLabel() = default;
  MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, Int_t qID);
  //: o2::MCCompLabel(trackID, eventID, srcID), mDetID(qID);
  Int_t getDetID() const { return mDetID; }

  ClassDefNV(MCLabel, 1);
};
} // namespace fit
} // namespace o2

#endif
