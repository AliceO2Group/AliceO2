// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_DATAPROCESSINGHEADER_H
#define FRAMEWORK_DATAPROCESSINGHEADER_H

#include "Headers/DataHeader.h"

#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <cassert>

namespace o2 {
namespace framework {

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
struct DataProcessingHeader : public header::BaseHeader
{
  // Required to do the lookup
  static const o2::header::HeaderType sHeaderType;
  static const uint32_t sVersion = 1;

  // allows DataHeader::SubSpecificationType to be used as generic type in the code
  using StartTime = uint64_t;
  using Duration = uint64_t;

  ///
  /// data start time
  ///
  StartTime startTime;

  ///
  /// data duration
  ///
  Duration duration;

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
  : BaseHeader(sizeof(DataProcessingHeader), sHeaderType, header::gSerializationMethodNone,sVersion),
    startTime(s),
    duration(d)
  {
  }

  DataProcessingHeader(const DataProcessingHeader&) = default;
  static const DataProcessingHeader* Get(const BaseHeader* baseHeader) {
    return (baseHeader->description==DataProcessingHeader::sHeaderType)?
    static_cast<const DataProcessingHeader*>(baseHeader):nullptr;
  }
};

} //namespace framework
} //namespace o2

#endif // FRAMEWORK_DATAPROCESSINGHEADER_H
