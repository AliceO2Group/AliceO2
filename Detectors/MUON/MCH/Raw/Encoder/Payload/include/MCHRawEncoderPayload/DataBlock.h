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

#ifndef O2_MCH_RAW_ENCODER_DATABLOCK_H
#define O2_MCH_RAW_ENCODER_DATABLOCK_H

#include <cstdint>
#include <vector>
#include <gsl/span>
#include <functional>
#include <iostream>
#include <optional>

namespace o2::mch::raw
{

/// @brief A lightweight struct to describe a MCH Raw Data Block
struct DataBlockHeader {
  uint32_t orbit;
  uint16_t bc;
  uint16_t solarId;
  uint64_t payloadSize;
};

/// @brief A DataBlock is a pair (DataBlockHeader,payload)
struct DataBlock {
  DataBlockHeader header;
  gsl::span<const std::byte> payload;
  uint64_t size() const
  {
    return sizeof(header) + payload.size();
  }
};

/// @brief a DataBlockRef is a pair (DataBlock,offset)
/// The offset is an offset into some _external_ buffer
struct DataBlockRef {
  DataBlock block;
  std::optional<uint64_t> offset;
};

/// Convert the header into bytes
void appendDataBlockHeader(std::vector<std::byte>& outBuffer, DataBlockHeader header);

/// Loop over a buffer, that should consist of (DataBlockHeader,payload) pairs
int forEachDataBlockRef(gsl::span<const std::byte> buffer,
                        std::function<void(DataBlockRef blockRef)> f);

/// Count the headers in the input buffer
int countHeaders(gsl::span<const std::byte> buffer);

std::ostream& operator<<(std::ostream& os, const DataBlockHeader& header);
std::ostream& operator<<(std::ostream& os, const DataBlockRef& ref);
std::ostream& operator<<(std::ostream& os, const DataBlock& block);

bool operator<(const DataBlockHeader& a, const DataBlockHeader& b);
bool operator<(const DataBlockRef& a, const DataBlockRef& rhs);

} // namespace o2::mch::raw

#endif
