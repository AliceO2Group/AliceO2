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

#include <gsl/span>
#include <iostream>
#include "IO.h"
#include <fmt/format.h>

namespace o2::mch::io::impl
{
template <typename T>
bool writeBinaryStruct(std::ostream& os,
                       gsl::span<const T> items)
{
  uint32_t nofItems = static_cast<uint32_t>(items.size());
  if (!nofItems) {
    return !os.bad();
  }
  writeNofItems(os, nofItems);
  os.write(reinterpret_cast<const char*>(items.data()), items.size_bytes());
  return !os.bad();
}
template <typename T>
bool readBinaryStruct(std::istream& in, std::vector<T>& items, const char* itemName)
{
  if (in.peek() == EOF) {
    return false;
  }
  // get the number of items
  int nitems = readNofItems(in, itemName);
  // get the items if any
  if (nitems > 0) {
    auto offset = items.size();
    items.resize(offset + nitems);
    in.read(reinterpret_cast<char*>(&items[offset]), nitems * sizeof(T));
    if (in.fail()) {
      throw std::length_error(fmt::format("invalid input : cannot read {} {}", nitems, itemName));
    }
  }
  return true;
}

} // namespace o2::mch::io::impl
