// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawEncoder/DataBlock.h"
#include <fmt/format.h>

namespace o2::mch::raw
{
void appendDataBlockHeader(std::vector<uint8_t>& outBuffer, DataBlockHeader header)
{
  gsl::span<uint8_t> ph(reinterpret_cast<uint8_t*>(&header), sizeof(ph));
  outBuffer.insert(outBuffer.end(), ph.begin(), ph.end());
}

int forEachDataBlockRef(gsl::span<const uint8_t> buffer,
                        std::function<void(DataBlockRef ref)> f)
{
  int index{0};
  int nheaders{0};
  DataBlockHeader header;
  while (index < buffer.size()) {
    memcpy(&header, &buffer[index], sizeof(header));
    nheaders++;
    if (f) {
      DataBlock block{header, buffer.subspan(index, header.payloadSize)};
      DataBlockRef ref{block, index};
      f(ref);
    }
    index += header.payloadSize + sizeof(header);
  }
  return nheaders;
}

int countHeaders(gsl::span<uint8_t> buffer)
{
  return forEachDataBlockRef(buffer, nullptr);
}

std::ostream& operator<<(std::ostream& os, const DataBlockHeader& header)
{
  os << fmt::format("ORB{:6d} BC{:4d} FEE{:4d} PAYLOADSIZE{:6d}",
                    header.orbit, header.bc, header.feeId, header.payloadSize);
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
  if (a.feeId < b.feeId) {
    return true;
  }
  if (a.feeId > b.feeId) {
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

} // namespace o2::mch::raw
