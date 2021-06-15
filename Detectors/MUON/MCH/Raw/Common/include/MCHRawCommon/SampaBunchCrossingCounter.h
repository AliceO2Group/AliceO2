// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_RAW_SAMPA_TIME_H
#define O2_MCH_RAW_SAMPA_TIME_H

#include "MCHRawCommon/DataFormats.h"
#include <tuple>

/** SampaBunchCrossingCounter is the internal Sampa counter at 40MHz,
  * and coded in 20 bits */

namespace o2::mch::raw
{

uint20_t sampaBunchCrossingCounter(uint32_t orbit, uint16_t bc, uint32_t firstOrbit);

std::tuple<uint32_t, uint16_t> orbitBC(uint20_t bunchCrossing, uint32_t firstOrbit);

} // namespace o2::mch::raw

#endif
