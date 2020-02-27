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

  /// \brief Comparison oparator, based on track, event and src Id
  /// \param another PHOS MCLabel
  /// \return result of comparison: same tracks or not
  bool operator==(const MCLabel& other) const { return compare(other) >= 0; }

  MCLabel& operator=(const MCLabel& other) = default;

  // /// \brief Add  deposited energy of a thack, do not check if track the same
  // /// \param another PHOS MCLabel
  void add(const MCLabel& other, float scale) { mEdep += other.mEdep * scale; }

  void scale(float s) { mEdep *= s; }

  float getEdep() const { return mEdep; }

  ClassDefNV(MCLabel, 1);
};
} // namespace phos
} // namespace o2

#endif
