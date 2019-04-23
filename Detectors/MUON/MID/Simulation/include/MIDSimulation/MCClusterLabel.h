// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDSimulation/MCClusterLabel.h
/// \brief  Label for MID clusters
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 April 2019

#ifndef O2_MID_MCCLUSTERLABEL_H
#define O2_MID_MCCLUSTERLABEL_H

#include <array>
#include "SimulationDataFormat/MCCompLabel.h"

namespace o2
{
namespace mid
{
class MCClusterLabel : public o2::MCCompLabel
{
 private:
  std::array<bool, 2> mFiredCathodes;

 public:
  MCClusterLabel() = default;
  MCClusterLabel(int trackID, int eventID, int srcID, bool isFiredBP, bool isFiredNBP);

  /// Sets flag stating if the bending plane was fired
  void setIsFiredBP(bool isFired) { mFiredCathodes[0] = isFired; };
  /// Gets flag stating if the bending plane was fired
  bool isFiredBP() const { return mFiredCathodes[0]; }

  /// Sets flag stating if the non-bending plane was fired
  void setIsFiredNBP(bool isFired) { mFiredCathodes[1] = isFired; };
  /// Gets flag stating if the non-bending plane was fired
  bool isFiredNBP() const { return mFiredCathodes[1]; }

  ClassDefNV(MCClusterLabel, 1);
}; // namespace mid
} // namespace mid
} // namespace o2

#endif
