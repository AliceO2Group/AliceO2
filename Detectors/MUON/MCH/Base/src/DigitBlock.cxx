// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHBase/DigitBlock.h"

namespace o2
{
namespace mch
{

std::ostream& operator<<(std::ostream& stream, const DigitStruct& digit)
{
  stream << "{uid = " << digit.uid << ", index = " << digit.index << ", adc = " << digit.adc << "}";
  return stream;
}

std::ostream& operator<<(std::ostream& stream, const DigitBlock& block)
{
  auto digit = reinterpret_cast<const DigitStruct*>(&block + 1);
  stream << "{header = " << block.header << ", fDigit[] = [";
  if (block.header.fNrecords > 0) {
    stream << digit[0];
  }
  for (uint32_t i = 1; i < block.header.fNrecords; ++i) {
    stream << ", " << digit[i];
  }
  stream << "]}";
  return stream;
}

bool DigitBlock::operator==(const DigitBlock& that) const
{
  auto digitA = reinterpret_cast<const DigitStruct*>(this + 1);
  auto digitB = reinterpret_cast<const DigitStruct*>(&that + 1);

  // First check if the blocks have the same header. If they do then check
  // if every digit is the same. In either case if we find a difference
  // return false.
  if (header != that.header) {
    return false;
  }
  for (uint32_t i = 0; i < header.fNrecords; ++i) {
    if (digitA[i] != digitB[i]) {
      return false;
    }
  }
  return true;
}

} // namespace mch
} // namespace o2
