// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DigitFileFormat.h"
#include <fmt/format.h>
#include <iostream>
#include <algorithm>
#include <string>
#include <stdexcept>

namespace o2::mch::io
{

std::array<DigitFileFormat, 4> digitFileFormats = {
  DigitFileFormat{2305844383603244847},
  DigitFileFormat{1224998065220435759},
  DigitFileFormat{63069292639436591},
  DigitFileFormat{1215990797246349103}};

std::ostream& operator<<(std::ostream& os, const DigitFileFormat& dff)
{
  os << fmt::format(
    "[ file version {} digit version {} size {} "
    "rof version {} size {} hasRof {} run2ids {} ] formatWord {}",
    dff.fileVersion,
    dff.digitVersion,
    dff.digitSize,
    dff.rofVersion,
    dff.rofSize,
    static_cast<bool>(dff.hasRof),
    static_cast<bool>(dff.run2ids),
    dff.format);
  return os;
}

bool operator==(const DigitFileFormat& dff1, const DigitFileFormat& dff2)
{
  return dff1.format == dff2.format;
}

bool operator!=(const DigitFileFormat& dff1, const DigitFileFormat& dff2)
{
  return !(dff1 == dff2);
}

/* Read the file format from the stream.
* 
* Every digit file should start with 8 bytes of format identifier.
*/
DigitFileFormat readDigitFileFormat(std::istream& in)
{
  uint64_t fileFormat{0};
  in.read(reinterpret_cast<char*>(&fileFormat), sizeof(uint64_t));
  if (in.gcount() < sizeof(DigitFileFormat)) {
    throw std::ios_base::failure("could not get a valid digit file format in this stream (too short)");
  }
  DigitFileFormat df{fileFormat};
  if (!isValid(df)) {
    throw std::ios_base::failure("could not get a valid digit file format in this stream");
  }
  return df;
}

bool isValid(DigitFileFormat dff)
{
  auto exists = std::find(digitFileFormats.begin(),
                          digitFileFormats.end(), dff);
  return exists != digitFileFormats.end();
}
} // namespace o2::mch::io
