// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_FRAMEWORK_OUTPUTOBJHEADER_H_
#define O2_FRAMEWORK_OUTPUTOBJHEADER_H_

#include "Headers/DataHeader.h"

using BaseHeader = o2::header::BaseHeader;

namespace o2::framework
{

/// Policy enum to determine OutputObj handling when writing.
enum OutputObjHandlingPolicy : unsigned int {
  AnalysisObject,
  QAObject,
  TransientObject,
  numPolicies
};

/// @struct OutputObjHeader
/// @brief O2 header for OutputObj metadata
struct OutputObjHeader : public BaseHeader {
  constexpr static const uint32_t sVersion = 1;
  constexpr static const o2::header::HeaderType sHeaderType = "OytObjMD";
  constexpr static const o2::header::SerializationMethod sSerializationMethod = o2::header::gSerializationMethodNone;
  OutputObjHandlingPolicy mPolicy;

  constexpr OutputObjHeader()
    : BaseHeader(sizeof(OutputObjHeader), sHeaderType, sSerializationMethod, sVersion),
      mPolicy{OutputObjHandlingPolicy::AnalysisObject} {}
  constexpr OutputObjHeader(OutputObjHandlingPolicy policy)
    : BaseHeader(sizeof(OutputObjHeader), sHeaderType, sSerializationMethod, sVersion),
      mPolicy{policy} {}
  constexpr OutputObjHeader(OutputObjHeader const&) = default;
};

} // namespace o2::framework

#endif // O2_FRAMEWORK_OUTPUTOBJHEADER_H_
