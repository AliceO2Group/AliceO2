// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MIDWorkflow/RawInputSpecHandler.h
/// \brief  Handler for raw data input specs
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 June 2021

#ifndef O2_MID_RAWINPUTSPECHANDLER_H
#define O2_MID_RAWINPUTSPECHANDLER_H

#include "Framework/InputSpec.h"
#include "Framework/Lifetime.h"
#include "Framework/ProcessingContext.h"
#include "Headers/DataHeader.h"

namespace o2
{
namespace mid
{
inline o2::framework::InputSpec getDiSTSTFSpec()
{
  return {"stdDist", "FLP", "DISTSUBTIMEFRAME", 0, o2::framework::Lifetime::Timeframe};
}

bool isDroppedTF(o2::framework::ProcessingContext& pc, o2::header::DataOrigin origin = o2::header::gDataOriginMID);

} // namespace mid
} // namespace o2

#endif //O2_MID_RAWINPUTSPECHANDLER_H
