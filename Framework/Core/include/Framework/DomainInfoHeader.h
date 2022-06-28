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
#ifndef O2_FRAMEWORK_DOMAININFOHEADER_H_
#define O2_FRAMEWORK_DOMAININFOHEADER_H_

#include "Headers/DataHeader.h"
#include "Framework/ChannelInfo.h"

#include <cstdint>
#include <memory>
#include <cassert>

namespace o2::framework
{

//__________________________________________________________________________________________________
/// @defgroup o2_dataflow_header The SourceInfo Header
/// @brief The DomainInfoHeader is used to pass information about the domain available data
///

//__________________________________________________________________________________________________
/// @struct DomainInfoHeader
/// @brief a BaseHeader with domain information from the source
///
///
/// @ingroup aliceo2_dataformats_dataheader
struct DomainInfoHeader : public header::BaseHeader {
  constexpr static const o2::header::HeaderType sHeaderType = "DmnInfo";
  static const uint32_t sVersion = 1;

  DomainInfoHeader()
    : BaseHeader(sizeof(DomainInfoHeader), sHeaderType, header::gSerializationMethodNone, sVersion)
  {
  }

  size_t oldestPossibleTimeslice = 0;

  DomainInfoHeader(const DomainInfoHeader&) = default;
  static const DomainInfoHeader* Get(const BaseHeader* baseHeader)
  {
    return (baseHeader->description == DomainInfoHeader::sHeaderType) ? static_cast<const DomainInfoHeader*>(baseHeader) : nullptr;
  }
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_DOMAININFOHEADER_H_
