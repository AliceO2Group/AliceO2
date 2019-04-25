// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_TRD_DIGIT_H_
#define ALICEO2_TRD_DIGIT_H_

#include <cstdint>
#include <vector>

namespace o2
{
namespace trd
{

class DigitIndex;

typedef std::uint16_t Digit;
typedef std::vector<Digit> DigitContainer;
typedef std::vector<DigitIndex> DigitIndexContainer;

} // namespace trd
} // namespace o2

#endif
