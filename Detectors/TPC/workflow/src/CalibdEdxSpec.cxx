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
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/NameConf.h"
#include "DataFormatsTPC/TrackTPC.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCalibration/Utils.h"
#include "Framework/Task.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/ConfigParamRegistry.h"
#include "TPCCalibration/CalibdEdx.h"
#include "TPCWorkflow/ProcessingHelpers.h"

using namespace o2::framework;

namespace o2::tpc
{

class CalibdEdxDevice : public Task
{
 public:
  void init(framework::InitContext& ic) final
  {
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

    mDumpToFile = ic.options().get<bool>("file-dump");
    auto field = ic.options().get<float>("field");

    if (field <= -10.f) {
      const auto inputGRP = o2::base::NameConf::getGRPFileName();
      const auto grp = o2::parameters::GRPObject::loadFrom(inputGRP);
      if (grp != nullptr) {
        field = 5.00668f * grp->getL3Current() / 30000.;
        LOGP(info, "Using GRP file to set the magnetic field to {} kG", field);
      }
    }

    mCalib = std::make_unique<CalibdEdx>(dEdxBins, mindEdx, maxdEdx, angularBins, fitSnp);
    mCalib->setApplyCuts(false);
    mCalib->setSectorFitThreshold(minEntriesSector);
    mCalib->set1DFitThreshold(minEntries1D);
    mCalib->set2DFitThreshold(minEntries2D);
    mCalib->setField(field);
    mCalib->setElectronCut(fitThreshold, fitPasses);
  }

  void run(ProcessingContext& pc) final
  {
    const auto tfcounter = o2::header::get<DataProcessingHeader*>(pc.inputs().get("tracks").header)->startTime;
    const auto tracks = pc.inputs().get<gsl::span<TrackTPC>>("tracks");

    LOGP(info, "Processing TF {} with {} tracks", tfcounter, tracks.size());
    mRunNumber = processing_helpers::getRunNumber(pc);
    mCalib->fill(tracks);
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
    const long timeEnd = o2::ccdb::CcdbObjectInfo::INFINITE_TIMESTAMP;

    info.setPath("TPC/Calib/dEdx");
    info.setStartValidityTimestamp(timeStart);
    info.setEndValidityTimestamp(timeEnd);

    auto md = info.getMetaData();
    md["runNumber"] = std::to_string(mRunNumber);
    info.setMetaData(md);

    auto image = o2::ccdb::CcdbApi::createObjectImage(&corr, &info);

    LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", info.getPath(), info.getFileName(), image->size(), info.getStartValidityTimestamp(), info.getEndValidityTimestamp());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "TPC_CalibdEdx", 0}, *image.get());
    output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "TPC_CalibdEdx", 0}, info);
  }

  bool mDumpToFile{};
  uint64_t mRunNumber{0}; ///< processed run number
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

      {"field", VariantType::Float, -100.f, {"magnetic field"}},
      {"file-dump", VariantType::Bool, false, {"directly dump calibration to file"}}}};
}

} // namespace o2::tpc
