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

#include "IO.h"
#include <stdexcept>
#include <fmt/format.h>
#include <iostream>

namespace o2::mch::io::impl
{
int readNofItems(std::istream& in, const char* itemName)
{
  int nitems(-1);
  in.read(reinterpret_cast<char*>(&nitems), sizeof(int));
  if (in.fail() || nitems < 0) {
    throw std::length_error(fmt::format("invalid input : cannot get number of {}", itemName));
  }
  return nitems;
}

void writeNofItems(std::ostream& out, uint32_t nofItems)
{
  out.write(reinterpret_cast<const char*>(&nofItems), sizeof(uint32_t));
}

int advance(std::istream& in, size_t itemByteSize, const char* itemName)
{
  if (in.peek() == EOF) {
    return -1;
  }
  // get the number of items
  int nitems = readNofItems(in, itemName);
  // move forward of n items
  auto current = in.tellg();
  in.seekg(current + static_cast<decltype(current)>(nitems * itemByteSize));
  return nitems;
}
} // namespace o2::mch::io::impl
