// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

// Declaration of a transient MC label class for PHOS

#ifndef ALICEO2_PHOS_MCLABEL_H_
#define ALICEO2_PHOS_MCLABEL_H_

#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace phos
{
class MCLabel : public o2::MCCompLabel
{
 private:
  float mEdep = 0; //deposited energy

 public:
  MCLabel() = default;
  MCLabel(Int_t trackID, Int_t eventID, Int_t srcID, bool fake, float edep) : o2::MCCompLabel(trackID, eventID, srcID, fake), mEdep(edep) {}
  float getEdep() const { return mEdep; }

  ClassDefNV(MCLabel, 1);
};
} // namespace phos
} // namespace o2

#endif
