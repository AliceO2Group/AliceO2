// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATACHUNK_H
#define FRAMEWORK_DATACHUNK_H

#include "MemoryResources/MemoryResources.h"

namespace o2
{
namespace framework
{
/// @class DataChunk A resizable buffer used with DPL's DataAllocator
/// DataChunk derives from std::vector with polymorphic allocator and forbids copying, the underlying
/// buffer is of type char and is through DPL and polymorphic memory resource directly allocated in the
/// message memory.
/// Since MessageContext returns the object by reference, the forbidden copy and assignment makes sure that
/// the code can not accidentally use a copy instead reference.
class DataChunk : public std::vector<char, o2::pmr::polymorphic_allocator<char>>
{
 public:
  // FIXME: want to have a general forwarding, but then the copy constructor is not deleted any more despite
  // it's declared deleted
  //template <typename... Args>
  //DataChunk(T&& arg, Args&&... args) : std::vector<char, o2::pmr::polymorphic_allocator<char>>(std::forward<Args>(args)...)
  //{
  //}

  // DataChunk is special and for the moment it's enough to declare the constructor with size and allocator
  DataChunk(size_t size, const o2::pmr::polymorphic_allocator<char>& allocator) : std::vector<char, o2::pmr::polymorphic_allocator<char>>(size, allocator)
  {
  }
  DataChunk(const DataChunk&) = delete;
  DataChunk& operator=(const DataChunk&) = delete;
  DataChunk(DataChunk&&) = default;
  DataChunk& operator=(DataChunk&&) = default;
};

} // namespace framework
} // namespace o2
#endif // FRAMEWORK_DATACHUNK_H
