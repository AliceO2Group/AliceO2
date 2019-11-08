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
#include <cstdint>
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
    mFile.open(filename.c_str(), std::ios::binary);
    if (!mFile.is_open()) {
      LOG(ERROR) << "Cannot open the " << filename << " file !";
      mState = 1;
      return;
    }

    auto headerOffset = ic.options().get<bool>("mid-raw-header-offset");
    mEncoder.setHeaderOffset(headerOffset);

    auto stop = [this]() {
      write();
      mFile.close();
    };
    ic.services().get<of::CallbackService>().set(of::CallbackService::Id::Stop, stop);
    mState = 0;
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
    write();
  }

 private:
  void write()
  {
    mFile.write(reinterpret_cast<const char*>(mEncoder.getBuffer().data()), mEncoder.getBufferSize());
    mEncoder.clear();
  }
  Encoder mEncoder{};
  std::ofstream mFile{};
  int mState{0};
};

framework::DataProcessorSpec getRawWriterSpec()
{
  std::vector<of::InputSpec> inputSpecs{of::InputSpec{"mid_data", "MID", "DATA"}, of::InputSpec{"mid_data_rof", "MID", "DATAROF"}, of::InputSpec{"mid_data_labels", "MID", "DATALABELS"}};

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
