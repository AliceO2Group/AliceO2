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

/// @file   TDCalibSpec.cxx
/// @brief  cxx file associated to TDCCalibSpec.h
/// @author luca.quaglia@cern.ch

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
#include "ZDCReconstruction/ZDCTDCParam.h"
#include "ZDCCalib/TDCCalibConfig.h"
#include "ZDCCalib/TDCCalibSpec.h"
#include "ZDCCalib/TDCCalibData.h"
#include "ZDCCalib/CalibParamZDC.h"

using namespace o2::framework;

namespace o2
{
namespace zdc
{

TDCCalibSpec::TDCCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

TDCCalibSpec::TDCCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void TDCCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  mTimer.Start(false);
}

void TDCCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ZDCTDCParam*>("tdccalib");
  pc.inputs().get<o2::zdc::TDCCalibConfig*>("tdccalibconfig");
}

void TDCCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIB", 0)) {
    // TDC calibration object
    auto* config = (const o2::zdc::ZDCTDCParam*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCParam(config);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "TDCCALIBCONFIG", 0)) {
    // TDC calibration configuration
    auto* config = (const o2::zdc::TDCCalibConfig*)obj;
    if (mVerbosity > DbgZero) {
      config->print();
    }
    mWorker.setTDCCalibConfig(config);
  }
}

void TDCCalibSpec::run(ProcessingContext& pc)
{
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mOutput = &(pc.outputs());
    mHistoFileMetaData = std::make_unique<o2::dataformats::FileMetaData>();
    mHistoFileMetaData->setDataTakingContext(pc.services().get<o2::framework::DataTakingContext>());
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
  }
  if (mRunStartTime == 0) {
    const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
    mRunStartTime = tinfo.creation; // approximate time in ms
    mRunNumber = tinfo.runNumber;
  }
  std::vector<InputSpec> filterHisto = {{"tdc_1dh", ConcreteDataTypeMatcher{"ZDC", "TDC_1DH"}, Lifetime::Timeframe}};
  for (auto const& inputRef : InputRecordWalker(pc.inputs(), filterHisto)) {
    auto const* dh = framework::DataRefUtils::getHeader<o2::header::DataHeader*>(inputRef);
    o2::dataformats::FlatHisto1D<float> histoView(pc.inputs().get<gsl::span<float>>(inputRef));
    mWorker.add(dh->subSpecification, histoView);
  }
  auto data = pc.inputs().get<TDCCalibData>("tdccalibdata");
  mWorker.process(data);
}

void TDCCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  mWorker.endOfRun();
  mTimer.Stop();
  sendOutput(ec);
  LOGF(info, "ZDC TDC calibration total timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1); // added by me
}

//________________________________________________________________
void TDCCalibSpec::sendOutput(o2::framework::EndOfStreamContext& ec)
{
  std::string fn = "ZDC_TDCCalib";
  o2::framework::DataAllocator& output = ec.outputs();

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getTDCParamUpd(); // new
  auto& info = mWorker.getCcdbObjectInfo();
  const auto& opt = CalibParamZDC::Instance();
  opt.updateCcdbObjectInfo(info);

  auto image = o2::ccdb::CcdbApi::createObjectImage<ZDCTDCParam>(&payload, &info); // new
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }
  mOutput->snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_TDCcalib", 0}, *image.get()); // vector<char>
  mOutput->snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_TDCcalib", 0}, info);         // root-serlized
  // TODO: reset the outputs once they are already sent (is it necessary?)
  // mWorker.init();

  if (opt.rootOutput == true) {
    mOutputDir = opt.outputDir;
    if (mOutputDir.compare("/dev/null")) {
      mHistoFileName = fmt::format("{}{}{}_{}.root", mOutputDir, mOutputDir.back() == '/' ? "" : "/", fn, mRunNumber);
      int rval = mWorker.write(mHistoFileName);
      if (rval) {
        LOG(error) << "Cannot create output file " << mHistoFileName;
        return;
      }
      std::string metaFileDir = opt.metaFileDir;
      if (metaFileDir.compare("/dev/null")) {
        mHistoFileMetaData->fillFileData(mHistoFileName);
        mHistoFileMetaData->type = "calib";
        mHistoFileMetaData->priority = "high";
        std::string metaFileNameTmp = metaFileDir + (metaFileDir.back() == '/' ? "" : "/") + fmt::format("{}_{}.tmp", fn, mRunNumber);
        std::string metaFileName = metaFileDir + (metaFileDir.back() == '/' ? "" : "/") + fmt::format("{}_{}.done", fn, mRunNumber);
        try {
          std::ofstream metaFileOut(metaFileNameTmp);
          metaFileOut << *mHistoFileMetaData.get();
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

framework::DataProcessorSpec getTDCCalibSpec()
{
  using device = o2::zdc::TDCCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("tdccalibconfig", "ZDC", "TDCCALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalibConfig.data())));
  inputs.emplace_back("tdccalib", "ZDC", "TDCCALIB", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(fmt::format("{}", o2::zdc::CCDBPathTDCCalib.data())));
  inputs.emplace_back("tdccalibdata", "ZDC", "TDCCALIBDATA", 0, Lifetime::Timeframe);
  inputs.emplace_back("tdc_1dh", ConcreteDataTypeMatcher{"ZDC", "TDC_1DH"}, Lifetime::Timeframe);

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDC_TDCcalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDC_TDCcalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-tdccalib",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
