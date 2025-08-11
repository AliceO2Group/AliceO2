// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_FRAMEWORK_DATACHUNK_H_
#define O2_FRAMEWORK_DATACHUNK_H_

#include <memory_resource>
#include <vector>

namespace o2::framework
{
/// @class DataChunk A resizable buffer used with DPL's DataAllocator
/// DataChunk derives from std::vector with polymorphic allocator and forbids copying, the underlying
/// buffer is of type char and is through DPL and polymorphic memory resource directly allocated in the
/// message memory.
/// Since MessageContext returns the object by reference, the forbidden copy and assignment makes sure that
/// the code can not accidentally use a copy instead reference.
class DataChunk : public std::vector<char, std::pmr::polymorphic_allocator<char>>
{
 public:
  // DataChunk is special and for the moment it's enough to declare the constructor with size and allocator
  DataChunk(size_t size, const std::pmr::polymorphic_allocator<char>& allocator) : std::vector<char, std::pmr::polymorphic_allocator<char>>(size, allocator)
  {
  }
  DataChunk(const DataChunk&) = delete;
  DataChunk& operator=(const DataChunk&) = delete;
  DataChunk(DataChunk&&) = default;
  DataChunk& operator=(DataChunk&&) = default;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATACHUNK_H_
