// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RawInputSpecHandler.cxx
/// \brief  Handler for raw data input specs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 June 2021

#include "MIDWorkflow/RawInputSpecHandler.h"

#include "Framework/DataRefUtils.h"
#include "Framework/InputRecordWalker.h"

namespace o2
{
namespace mid
{

bool isDroppedTF(o2::framework::ProcessingContext& pc, o2::header::DataOrigin origin)
{
  /// Tests it the TF was dropped
  std::vector<o2::framework::InputSpec> dummy{o2::framework::InputSpec{"dummy", o2::framework::ConcreteDataMatcher{origin, o2::header::gDataDescriptionRawData, 0xDEADBEEF}}};
  for (const auto& ref : o2::framework::InputRecordWalker(pc.inputs(), dummy)) {
    const auto dh = o2::framework::DataRefUtils::getHeader<o2::header::DataHeader*>(ref);
    if (dh->payloadSize == 0) {
      // LOGP(WARNING, "Found input [{}/{}/{:#x}] TF#{} 1st_orbit:{} Payload {} : assuming no payload for all links in this TF",
      //      dh->dataOrigin.str, dh->dataDescription.str, dh->subSpecification, dh->tfCounter, dh->firstTForbit, dh->payloadSize);
      return true;
    }
  }
  return false;
}
} // namespace mid
} // namespace o2
