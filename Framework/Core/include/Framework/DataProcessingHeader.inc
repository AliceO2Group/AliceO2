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
#ifndef O2_FRAMEWORK_DATAPROCESSINGHEADER_INC_
#define O2_FRAMEWORK_DATAPROCESSINGHEADER_INC_

#include <chrono>

namespace o2::framework
{
  uint64_t DataProcessingHeader::getCreationTime()
  {
    auto now = std::chrono::steady_clock::now();
    return ((uint64_t)std::chrono::duration<double, std::milli>(now.time_since_epoch()).count()) | DUMMY_CREATION_TIME_OFFSET;
  }

  DataProcessingHeader::DataProcessingHeader()
    : DataProcessingHeader(0, 0)
  {
  }

  DataProcessingHeader::DataProcessingHeader(StartTime s)
    : DataProcessingHeader(s, 0)
  {
  }

  DataProcessingHeader::DataProcessingHeader(StartTime s, Duration d)
    : BaseHeader(sizeof(DataProcessingHeader), sHeaderType, header::gSerializationMethodNone, sVersion),
      startTime(s),
      duration(d),
      creation(getCreationTime())
  {
  }

  DataProcessingHeader::DataProcessingHeader(StartTime s, Duration d, CreationTime t)
    : BaseHeader(sizeof(DataProcessingHeader), sHeaderType, header::gSerializationMethodNone, sVersion),
      startTime(s),
      duration(d),
      creation(t)
  {
  }

  static const DataProcessingHeader* Get(const header::BaseHeader* baseHeader)
  {
    return (baseHeader->description == DataProcessingHeader::sHeaderType) ? static_cast<const DataProcessingHeader*>(baseHeader) : nullptr;
  }
} // namespace o2::framework

#endif
