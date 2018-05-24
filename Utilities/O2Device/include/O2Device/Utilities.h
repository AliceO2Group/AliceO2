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

/// @brief standalone tools to interact with the O2 data modl
///
/// @author Mikolaj Krzewicki, mkrzewic@cern.ch

#ifndef ALICEO2_BASE_O2DEVICE_UTILITIES_
#define ALICEO2_BASE_O2DEVICE_UTILITIES_

#include "MemoryResources/MemoryResources.h"
#include "Headers/DataHeader.h"
#include <utility>

namespace o2
{
namespace Base
{

using O2Message = FairMQParts;

//__________________________________________________________________________________________________
// AddDataBlock for generic (compatible) containers, that is contiguous containers using the pmr allocator
template <typename ContainerT, typename std::enable_if<!std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool AddDataBlock(O2Message& parts, o2::header::Stack&& inputStack, ContainerT&& inputData, o2::memoryResources::FairMQMemoryResource* targetResource = nullptr)
{
  using std::move;
  auto dataMessage = getMessage(move(inputData), targetResource);
  return AddDataBlock(parts, move(inputStack), move(dataMessage), targetResource);
  return true;
}

//__________________________________________________________________________________________________
// AddDataBlock for data already wrapped in FairMQMessagePtr
// note: since we cannot partially specialize function templates, use SFINAE here instead
template <typename ContainerT, typename std::enable_if<std::is_same<ContainerT, FairMQMessagePtr>::value, int>::type = 0>
bool AddDataBlock(O2Message& parts, o2::header::Stack&& inputStack, ContainerT&& dataMessage, o2::memoryResources::FairMQMemoryResource* targetResource = nullptr)
{
  using std::move;
  auto headerMessage = getMessage(move(inputStack), targetResource);

  parts.AddPart(move(headerMessage));
  parts.AddPart(move(dataMessage));

  return true;
}

}; //namespace o2
}; //namespace Base

#endif
