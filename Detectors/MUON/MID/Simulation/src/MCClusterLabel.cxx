// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Simulation/src/MCClusterLabel.cxx
/// \brief  Implementation of MC label for MID clusters
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   23 April 2019

#include "MIDSimulation/MCClusterLabel.h"

ClassImp(o2::mid::MCClusterLabel);

namespace o2
{
namespace mid
{

MCClusterLabel::MCClusterLabel(int trackID, int eventID, int srcID, bool isFiredBP, bool isFiredNBP) : o2::MCCompLabel(trackID, eventID, srcID, false)
{
  /// Constructor
  setIsFiredBP(isFiredBP);
  setIsFiredNBP(isFiredNBP);
}

} // namespace mid
} // namespace o2
