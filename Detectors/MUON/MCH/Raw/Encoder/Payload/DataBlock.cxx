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

#include "MCHRawEncoderPayload/DataBlock.h"
#include <fmt/format.h>

namespace o2::mch::raw
{
void appendDataBlockHeader(std::vector<std::byte>& outBuffer, DataBlockHeader header)
{
  gsl::span<std::byte> ph(reinterpret_cast<std::byte*>(&header), sizeof(ph));
  outBuffer.insert(outBuffer.end(), ph.begin(), ph.end());
}

int forEachDataBlockRef(gsl::span<const std::byte> buffer,
                        std::function<void(DataBlockRef ref)> f)
{
  int index{0};
  int nheaders{0};
  DataBlockHeader header;
  while (index < buffer.size()) {
    memcpy(&header, &buffer[index], sizeof(header));
    nheaders++;
    if (f) {
      DataBlock block{header, buffer.subspan(index + sizeof(header), header.payloadSize)};
      DataBlockRef ref{block, index};
      f(ref);
    }
    index += header.payloadSize + sizeof(header);
  }
  return nheaders;
}

int countHeaders(gsl::span<const std::byte> buffer)
{
  return forEachDataBlockRef(buffer, nullptr);
}

std::ostream& operator<<(std::ostream& os, const DataBlockHeader& header)
{
  os << fmt::format("ORB{:6d} BC{:4d} SOLAR{:4d} PAYLOADSIZE{:6d}",
                    header.orbit, header.bc, header.solarId, header.payloadSize);
  return os;
}

std::ostream& operator<<(std::ostream& os, const DataBlock& block)
{
  os << block.header;
  os << fmt::format(" SIZE {:8d}", block.size());
  return os;
}

std::ostream& operator<<(std::ostream& os, const DataBlockRef& ref)
{
  os << ref.block;
  if (ref.offset.has_value()) {
    os << fmt::format(" OFFSET {:8d}", ref.offset.value());
  }
  return os;
}

bool operator<(const DataBlockHeader& a, const DataBlockHeader& b)
{
  if (a.solarId < b.solarId) {
    return true;
  }
  if (a.solarId > b.solarId) {
    return false;
  }
  if (a.orbit < b.orbit) {
    return true;
  }
  if (a.orbit > b.orbit) {
    return false;
  }
  return (a.bc < b.bc);
  // if (a.bc < b.bc) {
  //   return true;
  // }
  // if (a.bc > b.bc) {
  //   return false;
  // }
};

bool operator<(const DataBlockRef& a, const DataBlockRef& b)
{
  return a.block.header < b.block.header;
}
} // namespace o2::mch::raw
