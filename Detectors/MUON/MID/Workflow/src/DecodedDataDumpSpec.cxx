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

/// \file   MID/Workflow/src/RawDumpSpec.cxx
/// \brief  Device to dump decoded raw data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   17 February 2022

#include "MIDWorkflow/RawDumpSpec.h"

#include <fstream>
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ControlService.h"
#include "Framework/Logger.h"
#include "Framework/Task.h"
#include "fmt/format.h"
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"

namespace o2
{
namespace mid
{

class RawDumpDeviceDPL
{
 public:
  void init(o2::framework::InitContext& ic)
  {
    auto outFilename = ic.options().get<std::string>("mid-dump-outfile");

    if (!outFilename.empty()) {
      mOutFile.open(outFilename.c_str());
    }
  }

  void
    run(o2::framework::ProcessingContext& pc)
  {

    auto data = pc.inputs().get<gsl::span<ROBoard>>("mid_decoded");
    auto dataROFs = pc.inputs().get<gsl::span<ROFRecord>>("mid_decoded_rof");
    std::stringstream ss;
    for (auto& rof : dataROFs) {
      ss << fmt::format("BCid: 0x{:x} Orbit: 0x{:x}  EvtType: {:d}", rof.interactionRecord.bc, rof.interactionRecord.orbit, static_cast<int>(rof.eventType)) << std::endl;
      for (auto colIt = data.begin() + rof.firstEntry, end = data.begin() + rof.getEndIndex(); colIt != end; ++colIt) {
        ss << *colIt << std::endl;
      }
    }
    if (mOutFile.is_open()) {
      mOutFile << ss.str();
    } else {
      LOG(info) << ss.str();
    }
  }

 private:
  std::ofstream mOutFile; /// Output file
};

framework::DataProcessorSpec getRawDumpSpec()
{
  std::vector<o2::framework::InputSpec> inputSpecs{
    o2::framework::InputSpec{"mid_decoded", header::gDataOriginMID, "DECODED", 0, o2::framework::Lifetime::Timeframe},
    o2::framework::InputSpec{"mid_decoded_rof", header::gDataOriginMID, "DECODEDROF", 0, o2::framework::Lifetime::Timeframe}};

  return o2::framework::DataProcessorSpec{
    "MIDRawDataDumper",
    {inputSpecs},
    {},
    o2::framework::AlgorithmSpec{o2::framework::adaptFromTask<RawDumpDeviceDPL>()},
    o2::framework::Options{{"mid-dump-outfile", o2::framework::VariantType::String, "", {"Dump output to file"}}}};
}

} // namespace mid
} // namespace o2
