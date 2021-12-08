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

/// \file CalibratordEdxSpec.cxx
/// \brief Workflow for time based dE/dx calibration.
/// \author Thiago Badaró <thiago.saramela@usp.br>

#include "TPCWorkflow/CalibratordEdxSpec.h"

#include <vector>
#include <memory>

// o2 includes
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/CalibratordEdx.h"
#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibratordEdxDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final
  {
    const auto slotLength = ic.options().get<int>("tf-per-slot");
    const auto maxDelay = ic.options().get<int>("max-delay");
    const auto minEntries = ic.options().get<int>("min-entries");

    const auto minEntriesSector = ic.options().get<int>("min-entries-sector");
    const auto minEntries1D = ic.options().get<int>("min-entries-1d");
    const auto minEntries2D = ic.options().get<int>("min-entries-2d");

    const auto dEdxBins = ic.options().get<int>("dedxbins");
    const auto mindEdx = ic.options().get<float>("min-dedx");
    const auto maxdEdx = ic.options().get<float>("max-dedx");
    const auto angularBins = ic.options().get<int>("angularbins");
    const auto maxTgl = ic.options().get<float>("max-tgl");
    const auto zBins = ic.options().get<int>("zbins");

    const auto dumpData = ic.options().get<bool>("file-dump");
    auto field = ic.options().get<float>("field");

    if (field <= -10.f) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp != nullptr) {
        field = 5.00668f * grp->getL3Current() / 30000.;
        LOGP(info, "Using GRP file to set the magnetic field to {} kG", field);
      }
    }

    mCalibrator = std::make_unique<tpc::CalibratordEdx>();
    mCalibrator->setHistParams(dEdxBins, mindEdx, maxdEdx, angularBins, maxTgl);
    mCalibrator->setApplyCuts(false);

    mCalibrator->setFitCuts({minEntriesSector, minEntries1D, minEntries2D});
    mCalibrator->setField(field);
    mCalibrator->setMinEntries(minEntries);

    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);

    if (dumpData) {
      mCalibrator->enableDebugOutput("calibratordEdx.root");
    }
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("tracks").header)->startTime;
    const auto tracks = pc.inputs().get<gsl::span<tpc::TrackTPC>>("tracks");

    LOGP(info, "Processing TF {} with {} tracks", tfcounter, tracks.size());
    mRunNumber = processing_helpers::getRunNumber(pc);
    mCalibrator->process(tfcounter, tracks);
    sendOutput(pc.outputs());

    const auto& infoVec = mCalibrator->getTFinterval();
    LOGP(info, "Created {} objects for TF {}", infoVec.size(), tfcounter);
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    constexpr calibration::TFType INFINITE_TF = 0xffffffffffffffff;
    mCalibrator->checkSlotsToFinalize(INFINITE_TF);
    sendOutput(eos.outputs());

    if (mCalibrator->hasDebugOutput()) {
      mCalibrator->finalizeDebugOutput();
    }
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    using clbUtils = o2::calibration::Utils;
    const auto& calibrations = mCalibrator->getCalibs();
    auto& intervals = mCalibrator->getTFinterval();
    const long timeEnd = 99999999999999;

    for (unsigned int i = 0; i < calibrations.size(); i++) {
      const auto& object = calibrations[i];
      o2::ccdb::CcdbObjectInfo info;
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &info);

      info.setPath("TPC/Calib/dEdx");
      info.setStartValidityTimestamp(intervals[i].first);
      info.setEndValidityTimestamp(timeEnd);

      auto md = info.getMetaData();
      md["runNumber"] = std::to_string(mRunNumber);
      info.setMetaData(md);

      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());

      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx", i}, info);         // root-serialized
    }
    mCalibrator->initOutput(); // empty the outputs after they are send
  }

  std::unique_ptr<CalibratordEdx> mCalibrator;
  uint64_t mRunNumber{0}; ///< processed run number
};

DataProcessorSpec getCalibratordEdxSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-calibrator-dEdx",
    Inputs{
      InputSpec{"tracks", "TPC", "MIPS"},
    },
    outputs,
    adaptFromTask<CalibratordEdxDevice>(),
    Options{
      {"tf-per-slot", VariantType::Int, 100, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::Int, 3, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 50, {"minimum entries per stack to fit a single time slot"}},

      {"min-entries-sector", VariantType::Int, 1000, {"bellow this number of entries per, stack higher dimensional fits will be perform only for GEM stacks types (IROC, OROC1, ...). The mean is still corrected for every stack"}},
      {"min-entries-1d", VariantType::Int, 500, {"minimum entries per stack to fit 1D correction"}},
      {"min-entries-2d", VariantType::Int, 2500, {"minimum entries per stack to fit 2D correction"}},

      {"dedxbins", VariantType::Int, 100, {"number of dEdx bins"}},
      {"min-dedx", VariantType::Float, 5.0f, {"minimum value for dEdx histograms"}},
      {"max-dedx", VariantType::Float, 100.0f, {"maximum value for dEdx histograms"}},
      {"angularbins", VariantType::Int, 18, {"number angular bins: Tgl and Snp"}},
      {"max-tgl", VariantType::Float, 1.5f, {"maximum absolute value for Tgl histograms"}},
      {"zbins", VariantType::Int, 20, {"number of Z bins. Not used for now"}},

      {"field", VariantType::Float, -100.f, {"magnetic field"}},
      {"file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
