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
#include "DetectorsBase/GRPGeomHelper.h"
#include "TPCBase/CDBInterface.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibratordEdxDevice : public Task
{
 public:
  CalibratordEdxDevice(std::shared_ptr<o2::base::GRPGeomRequest> req) : mCCDBRequest(req) {}
  void init(framework::InitContext& ic) final
  {
    o2::base::GRPGeomHelper::instance().setRequest(mCCDBRequest);
    const auto slotLength = ic.options().get<uint32_t>("tf-per-slot");
    const auto maxDelay = ic.options().get<uint32_t>("max-delay");
    const auto minEntries = ic.options().get<int>("min-entries");

    const auto minEntriesSector = ic.options().get<int>("min-entries-sector");
    const auto minEntries1D = ic.options().get<int>("min-entries-1d");
    const auto minEntries2D = ic.options().get<int>("min-entries-2d");
    const auto fitPasses = ic.options().get<int>("fit-passes");
    const auto fitThreshold = ic.options().get<float>("fit-threshold");

    const auto dEdxBins = ic.options().get<int>("dedxbins");
    const auto mindEdx = ic.options().get<float>("min-dedx");
    const auto maxdEdx = ic.options().get<float>("max-dedx");
    const auto angularBins = ic.options().get<int>("angularbins");
    const auto fitSnp = ic.options().get<bool>("fit-snp");

    const auto dumpData = ic.options().get<bool>("file-dump");

    mCalibrator = std::make_unique<tpc::CalibratordEdx>();
    mCalibrator->setHistParams(dEdxBins, mindEdx, maxdEdx, angularBins, fitSnp);
    mCalibrator->setApplyCuts(false);
    mCalibrator->setFitThresholds(minEntriesSector, minEntries1D, minEntries2D);
    mCalibrator->setMinEntries(minEntries);
    mCalibrator->setSlotLength(slotLength);
    mCalibrator->setMaxSlotsDelay(maxDelay);
    mCalibrator->setElectronCut({fitThreshold, fitPasses});

    if (dumpData) {
      mCalibrator->enableDebugOutput("calibratordEdx.root");
    }
  }

  void finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj) final
  {
    o2::base::GRPGeomHelper::instance().finaliseCCDB(matcher, obj);
  }

  void run(ProcessingContext& pc) final
  {
    o2::base::GRPGeomHelper::instance().checkUpdates(pc);
    const auto tracks = pc.inputs().get<gsl::span<tpc::TrackTPC>>("tracks");
    o2::base::TFIDInfoHelper::fillTFIDInfo(pc, mCalibrator->getCurrentTFInfo());
    LOGP(detail, "Processing TF {} with {} tracks", mCalibrator->getCurrentTFInfo().tfCounter, tracks.size());
    mRunNumber = mCalibrator->getCurrentTFInfo().runNumber;
    mCalibrator->process(tracks);
    sendOutput(pc.outputs());

    const auto& infoVec = mCalibrator->getTFinterval();
    LOGP(detail, "Created {} objects for TF {}", infoVec.size(), mCalibrator->getCurrentTFInfo().tfCounter);
  }

  void endOfStream(EndOfStreamContext& eos) final
  {
    LOGP(info, "Finalizing calibration");
    mCalibrator->checkSlotsToFinalize(o2::calibration::INFINITE_TF);
    sendOutput(eos.outputs());

    if (mCalibrator->hasDebugOutput()) {
      mCalibrator->finalizeDebugOutput();
    }
  }

 private:
  void sendOutput(DataAllocator& output)
  {
    const auto& calibrations = mCalibrator->getCalibs();
    const auto& intervals = mCalibrator->getTimeIntervals();

    assert(calibrations.size() == intervals.size());
    for (unsigned int i = 0; i < calibrations.size(); i++) {
      const auto& object = calibrations[i];
      o2::ccdb::CcdbObjectInfo info(CDBTypeMap.at(CDBType::CalTimeGain), std::string{}, std::string{}, std::map<std::string, std::string>{{"runNumber", std::to_string(mRunNumber)}}, intervals[i].first, intervals[i].second + 1);
      auto image = o2::ccdb::CcdbApi::createObjectImage(&object, &info);
      LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx", i}, *image.get()); // vector<char>
      output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx", i}, info);         // root-serialized
    }
    mCalibrator->initOutput(); // empty the outputs after they are send
  }

  std::unique_ptr<CalibratordEdx> mCalibrator;
  std::shared_ptr<o2::base::GRPGeomRequest> mCCDBRequest;
  uint64_t mRunNumber{0}; ///< processed run number
};

DataProcessorSpec getCalibratordEdxSpec()
{
  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx"}, Lifetime::Sporadic);
  std::vector<InputSpec> inputs{{"tracks", "TPC", "MIPS"}};
  auto ccdbRequest = std::make_shared<o2::base::GRPGeomRequest>(true,                           // orbitResetTime
                                                                true,                           // GRPECS=true
                                                                false,                          // GRPLHCIF
                                                                true,                           // GRPMagField
                                                                false,                          // askMatLUT
                                                                o2::base::GRPGeomRequest::None, // geometry
                                                                inputs,
                                                                true,
                                                                true);
  return DataProcessorSpec{
    "tpc-calibrator-dEdx",
    inputs,
    outputs,
    adaptFromTask<CalibratordEdxDevice>(ccdbRequest),
    Options{
      {"tf-per-slot", VariantType::UInt32, 6000u, {"number of TFs per calibration time slot"}},
      {"max-delay", VariantType::UInt32, 10u, {"number of slots in past to consider"}},
      {"min-entries", VariantType::Int, 10000, {"minimum entries per stack to fit a single time slot"}},

      {"min-entries-sector", VariantType::Int, 1000, {"min entries per GEM stack to enable sector by sector correction. Below this value we only perform one fit per ROC type (IROC, OROC1, ...; no side nor sector information)."}},
      {"min-entries-1d", VariantType::Int, 10000, {"minimum entries per stack to fit 1D correction"}},
      {"min-entries-2d", VariantType::Int, 50000, {"minimum entries per stack to fit 2D correction"}},
      {"fit-passes", VariantType::Int, 3, {"number of fit iterations"}},
      {"fit-threshold", VariantType::Float, 0.2f, {"dEdx width around the MIP peak used in the fit"}},

      {"dedxbins", VariantType::Int, 60, {"number of dEdx bins"}},
      {"min-dedx", VariantType::Float, 20.0f, {"minimum value for dEdx histograms"}},
      {"max-dedx", VariantType::Float, 90.0f, {"maximum value for dEdx histograms"}},
      {"angularbins", VariantType::Int, 36, {"number of angular bins: Tgl and Snp"}},
      {"fit-snp", VariantType::Bool, false, {"enable Snp correction"}},

      {"file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
