// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/Workflow/src/RawWriterSpec.cxx
/// \brief  Digits to raw converter spec for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 October 2019

#include "MIDWorkflow/RawWriterSpec.h"

#include <fstream>
#include <gsl/gsl>
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/DataRefUtils.h"
#include "Framework/Logger.h"
#include "Framework/Output.h"
#include "Framework/Task.h"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ColumnData.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/Encoder.h"
#include "MIDRaw/FEEIdConfig.h"

namespace of = o2::framework;

namespace o2
{
namespace mid
{
class RawWriterDeviceDPL
{
 public:
  void init(o2::framework::InitContext& ic)
  {
    auto filename = ic.options().get<std::string>("mid-raw-outfile");
    mEncoder.init(filename.c_str());

    auto stop = [this]() {
      mEncoder.finalize();
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);

    // Write basic config files to be used with raw data reader workflow
    auto cfgFilename = filename;
    auto pos = filename.find_last_of('.');
    if (pos != std::string::npos) {
      cfgFilename.erase(pos);
    }
    cfgFilename += ".cfg";
    std::ofstream outCfgFile(cfgFilename.c_str());
    outCfgFile << "[defaults]\n";
    outCfgFile << "dataOrigin = " << header::gDataOriginMID.as<std::string>() << "\n\n";
    outCfgFile << "[input-file]\n";
    outCfgFile << "filePath = " << filename << "\n";
    outCfgFile.close();
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get("mid_data");
    gsl::span<const ColumnData> data = of::DataRefUtils::as<const ColumnData>(msg);

    auto msgROF = pc.inputs().get("mid_data_rof");
    gsl::span<const ROFRecord> rofRecords = of::DataRefUtils::as<const ROFRecord>(msgROF);

    for (auto& rofRecord : rofRecords) {
      auto eventData = data.subspan(rofRecord.firstEntry, rofRecord.nEntries);
      mEncoder.process(eventData, rofRecord.interactionRecord, rofRecord.eventType);
    }
  }

 private:
  Encoder mEncoder{};
};

framework::DataProcessorSpec getRawWriterSpec()
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_data", header::gDataOriginMID, "DATA"}, of::InputSpec{"mid_data_rof", header::gDataOriginMID, "DATAROF"}, of::InputSpec{"mid_data_labels", header::gDataOriginMID, "DATALABELS"}};

  return of::DataProcessorSpec{
    "MIDRawWriter",
    inputSpecs,
    of::Outputs{},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::RawWriterDeviceDPL>()},
    of::Options{
      {"mid-raw-outfile", of::VariantType::String, "mid_raw.dat", {"Raw output file name"}},
      {"mid-raw-header-offset", of::VariantType::Bool, false, {"Header offset in bytes"}}}};
}
} // namespace mid
} // namespace o2
