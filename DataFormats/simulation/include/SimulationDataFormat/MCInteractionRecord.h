// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @brief  Simulated interaction record

#ifndef ALICEO2_MCINTERACTIONRECORD_H
#define ALICEO2_MCINTERACTIONRECORD_H

#include <Rtypes.h>
#include <iostream>

namespace o2
{
struct MCInteractionRecord {
  double timeNS = 0.; ///< time in NANOSECONDS from start of run (period=0, orbit=0)
  int bc = 0;         ///< bunch crossing ID of interaction
  int orbit = 0;      ///< LHC orbit
  int period = 0;     ///< LHC period since beginning of run (if >0 -> time precision loss)

  MCInteractionRecord(double tNS, int bcr = 0, int orb = 0, int per = 0) : timeNS(tNS), bc(bcr), orbit(orb), period(per)
  {
  }

  void print()
  {
    std::cout << "BCid: " << bc << " Orbit: " << orbit << " Period: " << period << " T(ns): " << timeNS << std::endl;
  }

  ClassDefNV(MCInteractionRecord, 1);
};
}

#endif
