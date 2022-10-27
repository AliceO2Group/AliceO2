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

/// @file   InterCalibSpec.cxx
/// @brief  ZDC intercalibration
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include <filesystem>
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "CCDB/CcdbApi.h"
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/CCDBParamSpec.h"
#include "Framework/DataRefUtils.h"
#include "Framework/DataTakingContext.h"
#include "Framework/InputRecordWalker.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "DataFormatsZDC/BCData.h"
#include "DataFormatsZDC/ChannelData.h"
#include "DataFormatsZDC/OrbitData.h"
#include "DataFormatsZDC/RecEvent.h"
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCReconstruction/RecoConfigZDC.h"
#include "ZDCReconstruction/ZDCEnergyParam.h"
#include "ZDCReconstruction/ZDCTowerParam.h"
#include "ZDCCalib/InterCalibSpec.h"
#include "ZDCCalib/CalibParamZDC.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

InterCalibSpec::InterCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

InterCalibSpec::InterCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void InterCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
}

void InterCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::InterCalibConfig*>("intercalibconfig");
  pc.inputs().get<o2::zdc::ZDCEnergyParam*>("energycalib");
  pc.inputs().get<o2::zdc::ZDCTowerParam*>("towercalib");
}

void InterCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "INTERCALIBCONFIG", 0)) {
    // InterCalib configuration
    auto* config = (const o2::zdc::InterCalibConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setInterCalibConfig(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "ENERGYCALIB", 0)) {
    // Current energy calibration
    auto* config = (const o2::zdc::ZDCEnergyParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setEnergyParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TOWERCALIB", 0)) {
    // Tower intercalibration
    auto* config = (const o2::zdc::ZDCTowerParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTowerParam(config);
  }
}

void InterCalibSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }
  if (mRunStartTime == 0) {
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    mRunStartTime = tinfo.creation; // approximate time in ms
    mRunNumber = tinfo.runNumber;
  }
  {
    std::vector<InputSpec> filterHisto = {{"inter_1dh", ConcreteDataTypeMatcher{"ZDC", "INTER_1DH"}, Lifetime::Timeframe}};
    for (auto const& inputRef : InputRecordWalker(pc.inputs(), filterHisto)) {
      auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
      o2::dataformats::FlatHisto1D<float> histoView(pc.inputs().get<gsl::span<float>>(inputRef));
      mWorker.add(dh->subSpecification, histoView);
    }
  }
  {
    std::vector<InputSpec> filterHisto = {{"inter_2dh", ConcreteDataTypeMatcher{"ZDC", "INTER_2DH"}, Lifetime::Timeframe}};
    for (auto const& inputRef : InputRecordWalker(pc.inputs(), filterHisto)) {
      auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
      o2::dataformats::FlatHisto2D<float> histoView(pc.inputs().get<gsl::span<float>>(inputRef));
      mWorker.add(dh->subSpecification, histoView);
    }
  }
  auto data = pc.inputs().get<InterCalibData>("intercalibdata");
  mWorker.process(data);
}

void InterCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  sendOutput(ec);
  LOGF(info, "ZDC Intercalibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
}

//________________________________________________________________
void InterCalibSpec::sendOutput(EndOfStreamContext& ec)
{
  std::string fn = "ZDC_interCalib";
  o2::framework::DataAllocator& output = ec.outputs();

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getTowerParamUpd();
  auto& info = mWorker.getCcdbObjectInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage<ZDCTowerParam>(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_Intercalib", 0}, *image.get()); // vector<char>
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_Intercalib", 0}, info);         // root-serialized
  // TODO: reset the outputs once they are already sent (is it necessary?)
  // mWorker.init();

  const auto& opt = CalibParamZDC::Instance();
  if (opt.rootOutput == true) {
    mOutputDir = opt.outputDir;
    if (mOutputDir.compare("/dev/null")) {
      mHistoFileName = fmt::format("{}{}{}_{}.root", mOutputDir, mOutputDir.back() == '/' ? "" : "/", fn, mRunNumber);
      int rval = mWorker.saveDebugHistos(mHistoFileName);
      if (rval) {
        LOG(error) << "Cannot create output file " << mHistoFileName;
        return;
      }
      std::string metaFileDir = opt.metaFileDir;
      if (metaFileDir.compare("/dev/null")) {
        std::unique_ptr<o2::dataformats::FileMetaData> histoFileMetaData;
        histoFileMetaData = std::make_unique<o2::dataformats::FileMetaData>();
        histoFileMetaData->setDataTakingContext(ec.services().get<DataTakingContext>());
        histoFileMetaData->fillFileData(mHistoFileName);
        histoFileMetaData->type = "calib";
        histoFileMetaData->priority = "high";
        std::string metaFileNameTmp = metaFileDir + (metaFileDir.back() == '/' ? "" : "/") + fmt::format("{}_{}.tmp", fn, mRunNumber);
        std::string metaFileName = metaFileDir + (metaFileDir.back() == '/' ? "" : "/") + fmt::format("{}_{}.done", fn, mRunNumber);
        try {
          std::ofstream metaFileOut(metaFileNameTmp);
          metaFileOut << *histoFileMetaData.get();
          metaFileOut.close();
          std::filesystem::rename(metaFileNameTmp, metaFileName);
        } catch (std::exception const& e) {
          LOG(error) << "Failed to store ZDC meta data file " << metaFileName << ", reason: " << e.what();
        }
        LOG(info) << "Stored metadata file " << metaFileName << ".done";
      } else {
        LOG(info) << "Did not store metafile as meta-dir=" << metaFileDir;
      }
    } else {
      LOG(warn) << "Do not create output file since output dir is " << mOutputDir;
    }
  }
}

framework::DataProcessorSpec getInterCalibSpec()
{
  using device = o2::zdc::InterCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("intercalibconfig", "ZDC", "INTERCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathInterCalibConfig.data()));
  inputs.emplace_back("energycalib", "ZDC", "ENERGYCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathEnergyCalib.data()));
  inputs.emplace_back("towercalib", "ZDC", "TOWERCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathTowerCalib.data()));
  inputs.emplace_back("intercalibdata", "ZDC", "INTERCALIBDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("inter_1dh", ConcreteDataTypeMatcher{"ZDC", "INTER_1DH"}, Lifetime::Timeframe);
  inputs.emplace_back("inter_2dh", ConcreteDataTypeMatcher{"ZDC", "INTER_2DH"}, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_Intercalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_Intercalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-towers",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
