// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/** @file HitTime.h
 * C++ Muon MCH hit time definition.
 * @author  Andrea Ferrero, CEA
 */

#ifndef ALICEO2_MCH_BASE_HIT_TIME_H_
#define ALICEO2_MCH_BASE_HIT_TIME_H_

#include <cinttypes>

namespace o2
{
namespace mch
{

struct HitTime {
  union {
    // default value
    uint64_t time = 0x0000000000000000;
    struct {                       ///
      uint32_t sampaTime : 10;     /// bit 0 to 9: sampa time
      uint32_t bunchCrossing : 20; /// bit 10 to 29: bunch crossing counter
      uint32_t orbit;              /// bit 32 to 63: orbit
    };                             ///
  };
};

} //namespace mch
} //namespace o2
#endif // ALICEO2_MCH_BASE_HIT_TIME_H_
