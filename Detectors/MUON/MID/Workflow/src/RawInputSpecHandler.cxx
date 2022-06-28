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

/// \file   MID/Workflow/src/RawInputSpecHandler.cxx
/// \brief  Handler for raw data input specs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 June 2021

#include "MIDWorkflow/RawInputSpecHandler.h"
#include "Framework/Logger.h"
#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"
#include "CommonUtils/VerbosityConfig.h"

namespace o2
{
namespace mid
{

bool isDroppedTF(o2::framework::ProcessingContext& pc, o2::header::DataOrigin origin)
{
  /// Tests it the TF was dropped
  static size_t contDeadBeef = 0; // number of times 0xDEADBEEF was seen continuously
  std::vector<o2::framework::InputSpec> dummy{o2::framework::InputSpec{"dummy", o2::framework::ConcreteDataMatcher{origin, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
  for (const auto& ref : o2::framework::InputRecordWalker(pc.inputs(), dummy)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    auto payloadSize = o2::framework::DataRefUtils::getPayloadSize(ref);
    if (payloadSize == 0) {
      auto maxWarn = o2::conf::VerbosityConfig::Instance().maxWarnDeadBeef;
      if (++contDeadBeef <= maxWarn) {
        LOGP(alarm, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF{}",
             dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, payloadSize,
             contDeadBeef == maxWarn ? fmt::format(". {} such inputs in row received, stopping reporting", contDeadBeef) : "");
      }
      return true;
    }
  }
  contDeadBeef = 0; // if good data, reset the counter
  return false;
}
} // namespace mid
} // namespace o2
