// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @copyright
/// © Copyright 2014 Copyright Holders of the ALICE O2 collaboration.
/// See https://aliceinfo.cern.ch/AliceO2 for details on the Copyright holders.
/// This software is distributed under the terms of the
/// GNU General Public License version 3 (GPL Version 3).
///
/// License text in a separate file.
///
/// In applying this license, CERN does not waive the privileges and immunities
/// granted to it by virtue of its status as an Intergovernmental Organization
/// or submit itself to any jurisdiction.

/// @brief standalone tools to interact with the O2 data model
///
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch

#ifndef ALICEO2_BASE_O2DEVICE_UTILITIES_
#define ALICEO2_BASE_O2DEVICE_UTILITIES_

#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

#include <utility>
#include <gsl/gsl>

namespace o2
{
namespace base
{

using O2Message = FairMQParts;

//__________________________________________________________________________________________________
// addDataBlock for generic (compatible) containers, that is contiguous containers using the pmr allocator
template <typename ContainerT, typename std::enable_if<!std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool addDataBlock(O2Message& parts, o2::header::Stack&& inputStack, ContainerT&& inputData, o2::pmr::FairMQMemoryResource* targetResource = nullptr)
{
  using std::forward;
  using std::move;

  auto headerMessage = getMessage(move(inputStack), targetResource);
  auto dataMessage = getMessage(forward<ContainerT>(inputData), targetResource);

  parts.AddPart(move(headerMessage));
  parts.AddPart(move(dataMessage));

  return true;
}

//__________________________________________________________________________________________________
// addDataBlock for data already wrapped in FairMQMessagePtr
// note: since we cannot partially specialize function templates, use SFINAE here instead
template <typename ContainerT, typename std::enable_if<std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool addDataBlock(O2Message& parts, o2::header::Stack&& inputStack, ContainerT&& dataMessage, o2::pmr::FairMQMemoryResource* targetResource = nullptr)
{
  using std::move;

  //make sure the payload size in DataHeader corresponds to message size
  using o2::header::DataHeader;
  DataHeader* dataHeader = const_cast<DataHeader*>(o2::header::get<DataHeader*>(inputStack.data()));
  dataHeader->payloadSize = dataMessage->GetSize();

  auto headerMessage = getMessage(move(inputStack), targetResource);

  parts.AddPart(move(headerMessage));
  parts.AddPart(move(dataMessage));

  return true;
}

namespace internal
{

template <typename I, typename F>
auto forEach(I begin, I end, F&& function)
{
  using span = gsl::span<const o2::byte>;
  using gsl::narrow_cast;
  for (auto it = begin; it != end; ++it) {
    o2::byte* headerBuffer{nullptr};
    span::index_type headerBufferSize{0};
    if (*it != nullptr) {
      headerBuffer = reinterpret_cast<o2::byte*>((*it)->GetData());
      headerBufferSize = narrow_cast<span::index_type>((*it)->GetSize());
    }
    ++it;
    o2::byte* dataBuffer{nullptr};
    span::index_type dataBufferSize{0};
    if (*it != nullptr) {
      dataBuffer = reinterpret_cast<o2::byte*>((*it)->GetData());
      dataBufferSize = narrow_cast<span::index_type>((*it)->GetSize());
    }

    // call the user provided function
    function(span{headerBuffer, headerBufferSize}, span{dataBuffer, dataBufferSize});
  }
  return std::move(function);
}
}; //namespace internal

/// Execute user code (e.g. a lambda) on each data block (header-payload pair)
/// returns the function (same as std::for_each)
template <typename F>
auto forEach(O2Message& parts, F&& function)
{
  if ((parts.Size() % 2) != 0) {
    throw std::invalid_argument(
      "number of parts in message not even (n%2 != 0), cannot be considered an O2 compliant message");
  }

  return internal::forEach(parts.begin(), parts.end(), std::forward<F>(function));
}

} // namespace base
} // namespace o2

#endif
