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

/// \file CalibdEdxSpec.cxx
/// \brief Workflow for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#include "TPCWorkflow/CalibdEdxSpec.h"

// o2 includes
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsParameters/GRPObject.h"
#include "CommonUtils/NameConf.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/CalibdEdx.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibdEdxDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final
  {
    const int minEntriesSector = ic.options().get<int>("min-entries-sector");
    const int minEntries1D = ic.options().get<int>("min-entries-1d");
    const int minEntries2D = ic.options().get<int>("min-entries-2d");

    const int dEdxBins = ic.options().get<int>("dedxbins");
    const int zBins = ic.options().get<int>("zbins");
    const int angularBins = ic.options().get<int>("angularbins");
    const float mindEdx = ic.options().get<float>("min-dedx");
    const float maxdEdx = ic.options().get<float>("max-dedx");

    mDumpToFile = ic.options().get<bool>("file-dump");
    float field = ic.options().get<float>("field");

    if (field <= -10.f) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp != nullptr) {
        field = 5.00668f * grp->getL3Current() / 30000.;
        LOGP(info, "Using GRP file to set the magnetic field to {} kG", field);
      }
    }

    mCalib = std::make_unique<CalibdEdx>(mindEdx, maxdEdx, dEdxBins, zBins, angularBins);
    mCalib->setApplyCuts(false);
    mCalib->setFitCuts({minEntriesSector, minEntries1D, minEntries2D});
    mCalib->setField(field);
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("tracks").header)->startTime;
    const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");

    LOGP(info, "Processing TF {} with {} tracks", tfcounter, tracks.size());

    mCalib->fill(tracks);
    // sendOutput(pc.outputs());
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    mCalib->finalize();
    mCalib->print();
    sendOutput(eos.outputs());

    if (mDumpToFile) {
      mCalib->getCalib().writeToFile("calibdEdx.root");
    }
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    using clbUtils = o2::calibration::Utils;
    const auto& corr = mCalib->getCalib();

    o2::ccdb::CcdbObjectInfo info;

    const auto now = std::chrono::system_clock::now();
    const long timeStart = std::chrono::duration_cast<std::chrono::milliseconds>(now.time_since_epoch()).count();
    const long timeEnd = 99999999999999;

    info.setPath("TPC/Calib/dEdx");
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(timeEnd);

    auto image = o2::ccdb::CcdbApi::createObjectImage(&corr, &info);

    LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx", 0}, info);
  }

  bool mDumpToFile{};
  std::unique_ptr<CalibdEdx> mCalib;
};

DataProcessorSpec getCalibdEdxSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-calib-dEdx",
    Inputs{
      InputSpec{"tracks", "TPC", "MIPS"},
    },
    outputs,
    adaptFromTask<CalibdEdxDevice>(),
    Options{
      {"min-entries-sector", VariantType::Int, 1000, {"bellow this number of entries per, stack higher dimensional fits will be perform only for GEM stacks types (IROC, OROC1, ...). The mean is still corrected for every stack"}},
      {"min-entries-1d", VariantType::Int, 500, {"minimum entries per stack to fit 1D correction"}},
      {"min-entries-2d", VariantType::Int, 2500, {"minimum entries per stack to fit 2D correction"}},

      {"dedxbins", VariantType::Int, 100, {"number of dEdx bins"}},
      {"zbins", VariantType::Int, 20, {"number of Z bins"}},
      {"angularbins", VariantType::Int, 18, {"number of bins for angular data, like Tgl and Snp"}},
      {"min-dedx", VariantType::Float, 5.0f, {"minimum value for the dEdx histograms"}},
      {"max-dedx", VariantType::Float, 100.0f, {"maximum value for the dEdx histograms"}},

      {"field", VariantType::Float, -100.f, {"magnetic field"}},
      {"file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
