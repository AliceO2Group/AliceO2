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

#pragma once

#include <iosfwd>
#include <cstdint>

namespace o2::mch::io::impl
{
int readNofItems(std::istream& in, const char* itemName);
void writeNofItems(std::ostream& out, uint32_t nofItems);
int advance(std::istream& in, size_t itemByteSize, const char* itemName);

} // namespace o2::mch::io::impl
