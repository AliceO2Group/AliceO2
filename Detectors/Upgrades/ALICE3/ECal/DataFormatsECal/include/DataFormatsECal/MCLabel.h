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

/// \file MCLabel.h
/// \brief MCLabel class to store MC truth info for ECal
///
/// \author Evgeny Kryshen <evgeny.kryshen@cern.ch>

#ifndef ALICEO2_ECAL_MCLABEL_H
#define ALICEO2_ECAL_MCLABEL_H

#include <SimulationDataFormat/MCCompLabel.h>

namespace o2
{
namespace ecal
{
class MCLabel : public o2::MCCompLabel
{
 public:
  MCLabel() = default;
  MCLabel(int trackID, int eventID, int srcID, bool fake, float edep) : o2::MCCompLabel(trackID, eventID, srcID, fake), mEdep(edep) {}
  float getEdep() const { return mEdep; }

 private:
  float mEdep = 0; // deposited energy

  ClassDefNV(MCLabel, 1);
};
} // namespace ecal
} // namespace o2

#endif // ALICEO2_ECAL_MCLABEL_H
