// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#pragma once

#include <cstdint>
#include <iosfwd>
#include <array>

namespace o2::mch::io
{

constexpr uint64_t TAG_DIGITS = 0x3F2F; // = 016175 = 0.1.6.1.7.5 = D.I.G.I.T.S

struct DigitFileFormat {
  union {
    uint64_t format = TAG_DIGITS;
    struct {
      uint64_t tag : 16;
      uint64_t fileVersion : 8;
      uint64_t reserved : 4;
      uint64_t digitVersion : 8;
      uint64_t digitSize : 8;
      uint64_t rofVersion : 8;
      uint64_t rofSize : 8;
      uint64_t hasRof : 1;
      uint64_t run2ids : 1;
    };
  };
};

extern std::array<DigitFileFormat, 4> digitFileFormats;

std::ostream& operator<<(std::ostream&, const DigitFileFormat&);

bool operator==(const DigitFileFormat& dff1, const DigitFileFormat& dff2);
bool operator!=(const DigitFileFormat& dff1, const DigitFileFormat& dff2);

DigitFileFormat readDigitFileFormat(std::istream& in);

bool isValid(DigitFileFormat dff);

} // namespace o2::mch::io
