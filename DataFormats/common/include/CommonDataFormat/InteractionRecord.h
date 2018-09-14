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
  double timeNS = 0.; ///< time in NANOSECONDS from start of run (period=0, orbit=0)
  int bc = 0;         ///< bunch crossing ID of interaction
  int orbit = 0;      ///< LHC orbit
  int period = 0;     ///< LHC period since beginning of run (if >0 -> time precision loss)

  InteractionRecord() = default;

  InteractionRecord(double tNS)
  {
    setFromNS(tNS);
  }

  InteractionRecord(int b, int orb, int per = 0) : bc(b), orbit(orb), period(per)
  {
    timeNS = bc2ns(bc, orbit, period);
  }

  void setFromNS(double ns)
  {
    timeNS = ns;
    bc = ns2bc(ns, orbit, period);
  }

  static double bc2ns(int bc, int orbit, int period)
  {
    double t = bc * o2::constants::lhc::LHCBunchSpacingNS + orbit * o2::constants::lhc::LHCOrbitNS;
    return period ? t + o2::constants::lhc::PeriodDurationNS : t;
  }

  static int ns2bc(double ns, int& orb, int& per)
  {
    per = ns / o2::constants::lhc::PeriodDurationNS;
    if (per) {
      ns -= per * o2::constants::lhc::PeriodDurationNS;
    }
    orb = ns / o2::constants::lhc::LHCOrbitNS;
    ns -= orb * o2::constants::lhc::LHCOrbitNS;
    return std::round(ns / o2::constants::lhc::LHCBunchSpacingNS);
  }

  void print() const;

  ClassDefNV(InteractionRecord, 1);
};
}

#endif
