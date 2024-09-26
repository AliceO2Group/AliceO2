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
#ifndef O2_FRAMEWORK_DATAPROCESSINGHEADER_H_
#define O2_FRAMEWORK_DATAPROCESSINGHEADER_H_

#include "Headers/DataHeader.h"

#include <cstdint>
#include <memory>
#include <cassert>
#include <chrono>

namespace o2::framework
{

//__________________________________________________________________________________________________
/// @defgroup o2_dataflow_header The DataFlow Header
/// @brief A descriptive information for data blocks handled by O2 Data Processing layer
///
/// @ingroup aliceo2_dataformat_primitives

//__________________________________________________________________________________________________
/// @struct DataFlowHeader
/// @brief a DataHeader with a time interval associated to it
///
/// Because the DataHeader is a generic entity, it does not have any time
/// attached to it, however the Data Processing layer does have an inherent
/// concept of time associated to each group of messages being processed,
/// therefore whenever some data enters the Data Processing layer it needs to
/// time information to be attached to it.
///
/// The information consists of two parts: start time id and duration
///
/// @ingroup aliceo2_dataformats_dataheader
struct DataProcessingHeader : public header::BaseHeader {
  static constexpr uint64_t DUMMY_CREATION_TIME_OFFSET = 0x8000000000000000;
  // The following flags are used to indicate the behavior of the data processing
  static constexpr int32_t KEEP_AT_EOS_FLAG = 1;

  /// We return some number of milliseconds, offsetting int by 0x8000000000000000
  /// to make sure we can understand when the dummy constructor of DataProcessingHeader was
  /// used without overriding it with an actual real time from epoch.
  /// This creation time is not meant to be used for anything but to understand the relative
  /// creation of messages in the flow. Notice that for the case DataProcessingHeader::creation
  /// has some particular meaning, we expect this function not to be used.
  static uint64_t getCreationTime()
  {
    auto now = std::chrono::steady_clock::now();
    return ((uint64_t)std::chrono::duration<double, std::milli>(now.time_since_epoch()).count()) | DUMMY_CREATION_TIME_OFFSET;
  }
  // Required to do the lookup
  constexpr static const o2::header::HeaderType sHeaderType = "DataFlow";
  static const uint32_t sVersion = 1;

  // allows DataHeader::SubSpecificationType to be used as generic type in the code
  using StartTime = uint64_t;
  using Duration = uint64_t;
  using CreationTime = uint64_t;

  ///
  /// data start time
  ///
  StartTime startTime;

  ///
  /// data duration
  ///
  Duration duration;

  CreationTime creation;

  //___NEVER MODIFY THE ABOVE
  //___NEW STUFF GOES BELOW

  //___the functions:
  DataProcessingHeader()
    : DataProcessingHeader(0, 0)
  {
  }

  DataProcessingHeader(StartTime s)
    : DataProcessingHeader(s, 0)
  {
  }

  DataProcessingHeader(StartTime s, Duration d)
    : BaseHeader(sizeof(DataProcessingHeader), sHeaderType, header::gSerializationMethodNone, sVersion),
      startTime(s),
      duration(d),
      creation(getCreationTime())
  {
  }

  DataProcessingHeader(StartTime s, Duration d, CreationTime t)
    : BaseHeader(sizeof(DataProcessingHeader), sHeaderType, header::gSerializationMethodNone, sVersion),
      startTime(s),
      duration(d),
      creation(t)
  {
  }

  DataProcessingHeader(const DataProcessingHeader&) = default;
  static const DataProcessingHeader* Get(const BaseHeader* baseHeader)
  {
    return (baseHeader->description == DataProcessingHeader::sHeaderType) ? static_cast<const DataProcessingHeader*>(baseHeader) : nullptr;
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DATAPROCESSINGHEADER_H_
