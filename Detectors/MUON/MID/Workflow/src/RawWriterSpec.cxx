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

/// \file   MID/Workflow/src/RawWriterSpec.cxx
/// \brief  Digits to raw converter spec for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   02 October 2019

#include "MIDWorkflow/RawWriterSpec.h"
#include <fstream>
#include <filesystem>
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
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"

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
    auto dirname = ic.options().get<std::string>("mid-raw-outdir");
    auto fileFor = ic.options().get<std::string>("file-for");
    if (!std::filesystem::exists(dirname)) {
      if (!std::filesystem::create_directories(dirname)) {
        LOG(fatal) << "could not create output directory " << dirname;
      } else {
        LOG(info) << "created output directory " << dirname;
      }
    }

    mEncoder.init(dirname.c_str(), fileFor.c_str());

    std::string inputGRP = o2::base::NameConf::getGRPFileName();
    std::unique_ptr<o2::parameters::GRPObject> grp{o2::parameters::GRPObject::loadFrom(inputGRP)};
    mEncoder.getWriter().setContinuousReadout(grp->isDetContinuousReadOut(o2::detectors::DetID::MID)); // must be set explicitly

    auto stop = [this]() {
      mEncoder.finalize();
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);

    // Write basic config files to be used with raw data reader workflow
    mEncoder.getWriter().writeConfFile("MID", "RAWDATA", o2::utils::Str::concat_string(dirname, '/', "MIDraw.cfg"));
  }

  void run(o2::framework::ProcessingContext& pc)
  {
    auto msg = pc.inputs().get("mid_data_mc");
    gsl::span<const ColumnData> data = of::DataRefUtils::as<const ColumnData>(msg);

    auto msgROF = pc.inputs().get("mid_data_mc_rof");
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
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_data_mc", header::gDataOriginMID, "DATAMC"}, of::InputSpec{"mid_data_mc_rof", header::gDataOriginMID, "DATAMCROF"}};

  return of::DataProcessorSpec{
    "MIDRawWriter",
    inputSpecs,
    of::Outputs{},
    of::AlgorithmSpec{of::adaptFromTask<o2::mid::RawWriterDeviceDPL>()},
    of::Options{
      {"mid-raw-outdir", of::VariantType::String, ".", {"Raw file output directory"}},
      {"file-for", of::VariantType::String, "all", {"single file per: all,flp,cruendpoint,link"}},
      {"mid-raw-header-offset", of::VariantType::Bool, false, {"Header offset in bytes"}}}};
}
} // namespace mid
} // namespace o2
