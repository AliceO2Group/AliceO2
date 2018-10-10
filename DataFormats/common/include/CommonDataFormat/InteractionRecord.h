// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Interaction record encoding BC, orbit, time

#ifndef ALICEO2_INTERACTIONRECORD_H
#define ALICEO2_INTERACTIONRECORD_H

#include <Rtypes.h>
#include <iosfwd>
#include <cmath>
#include "CommonConstants/LHCConstants.h"

namespace o2
{
struct InteractionRecord {
  double timeNS = 0.; ///< time in NANOSECONDS from start of run (orbit=0)
  int bc = 0;         ///< bunch crossing ID of interaction
  unsigned int orbit = 0; ///< LHC orbit

  InteractionRecord() = default;

  InteractionRecord(double tNS)
  {
    setFromNS(tNS);
  }

  InteractionRecord(int b, unsigned int orb) : bc(b), orbit(orb)
  {
    timeNS = bc2ns(bc, orbit);
  }

  void setFromNS(double ns)
  {
    timeNS = ns;
    bc = ns2bc(ns, orbit);
  }

  static double bc2ns(int bc, unsigned int orbit)
  {
    return bc * o2::constants::lhc::LHCBunchSpacingNS + orbit * o2::constants::lhc::LHCOrbitNS;
  }

  static int ns2bc(double ns, unsigned int& orb)
  {
    orb = ns > 0 ? ns / o2::constants::lhc::LHCOrbitNS : 0;
    ns -= orb * o2::constants::lhc::LHCOrbitNS;
    return std::round(ns / o2::constants::lhc::LHCBunchSpacingNS);
  }

  void print() const;

  ClassDefNV(InteractionRecord, 2);
};
}

#endif
