// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef FRAMEWORK_SOURCEINFOHEADER_H
#define FRAMEWORK_SOURCEINFOHEADER_H

#include "Headers/DataHeader.h"
#include "Framework/ChannelInfo.h"

#include <cstdint>
#include <cstdio>
#include <memory>
#include <cassert>

namespace o2
{
namespace framework
{

//__________________________________________________________________________________________________
/// @defgroup o2_dataflow_header The SourceInfo Header
/// @brief A descriptive information for data blocks handled by O2 Data Processing layer
///
/// @ingroup aliceo2_dataformat_primitives

//__________________________________________________________________________________________________
/// @struct SourceInfoHeader
/// @brief a BaseHeader with state information from the source
///
/// This header can be associated to a given O2 message to notify the receiving side
/// what is the state of the preceding dataprocessor in the dag associated to a given
/// channel.
///
/// @ingroup aliceo2_dataformats_dataheader
struct SourceInfoHeader : public header::BaseHeader {
  constexpr static const o2::header::HeaderType sHeaderType = "SrcInfo";
  static const uint32_t sVersion = 1;

  SourceInfoHeader()
    : BaseHeader(sizeof(SourceInfoHeader), sHeaderType, header::gSerializationMethodNone, sVersion),
      state{InputChannelState::Running} {}

  InputChannelState state;

  SourceInfoHeader(const SourceInfoHeader&) = default;
  static const SourceInfoHeader* Get(const BaseHeader* baseHeader)
  {
    return (baseHeader->description == SourceInfoHeader::sHeaderType) ? static_cast<const SourceInfoHeader*>(baseHeader) : nullptr;
  }
};

} // namespace framework
} // namespace o2

#endif // FRAMEWORK_SOURCEINFOHEADER_H
