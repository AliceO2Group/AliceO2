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

/// @file   BaselineCalibSpec.cxx
/// @brief  ZDC baseline calibration
/// @author pietro.cortese@cern.ch

#include <iostream>
#include <vector>
#include <string>
#include <gsl/span>
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
#include "ZDCBase/ModuleConfig.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/MemFileHelper.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CCDBTimeStampUtils.h"
#include "ZDCCalib/BaselineCalibSpec.h"
#include "ZDCCalib/BaselineCalibData.h"
#include "ZDCCalib/CalibParamZDC.h"
using namespace o2::framework;

namespace o2
{
namespace zdc
{

BaselineCalibSpec::BaselineCalibSpec()
{
  mTimer.Stop();
  mTimer.Reset();
}

BaselineCalibSpec::BaselineCalibSpec(const int verbosity) : mVerbosity(verbosity)
{
  mTimer.Stop();
  mTimer.Reset();
}

void BaselineCalibSpec::init(o2::framework::InitContext& ic)
{
  mVerbosity = ic.options().get<int>("verbosity-level");
  mWorker.setVerbosity(mVerbosity);
  const auto& opt = CalibParamZDC::Instance();
  mCTimeMod = opt.mCTimeMod;
  LOG(info) << "BaselineCalibSpec::init: mVerbosity=" << mVerbosity << " mCTimeMod=" << mCTimeMod;
  mProcessed = 0;
}

void BaselineCalibSpec::updateTimeDependentParams(ProcessingContext& pc)
{
  // we call these methods just to trigger finaliseCCDB callback
  pc.inputs().get<o2::zdc::ModuleConfig*>("moduleconfig");
  pc.inputs().get<o2::zdc::BaselineCalibConfig*>("calibconfig");
}

void BaselineCalibSpec::finaliseCCDB(o2::framework::ConcreteDataMatcher& matcher, void* obj)
{
  if (matcher == ConcreteDataMatcher("ZDC", "MODULECONFIG", 0)) {
    mWorker.setModuleConfig((const o2::zdc::ModuleConfig*)obj);
  }
  if (matcher == ConcreteDataMatcher("ZDC", "BASECALIBCONFIG", 0)) {
    mWorker.setConfig((const o2::zdc::BaselineCalibConfig*)obj);
  }
}

void BaselineCalibSpec::run(ProcessingContext& pc)
{
  const auto& tinfo = pc.services().get<o2::framework::TimingInfo>();
  const auto* dh = DataRefUtils::getHeader<o2::header::DataHeader*>(pc.inputs().getFirstValid(true));

#ifdef O2_ZDC_DEBUG
  LOG(info) << "BaselineCalibSpec::run mInitialized=" << mInitialized << " run(ti:dh)=" << tinfo.runNumber << ":" << dh->runNumber << " mRunStartTime=" << mRunStartTime << " mProcessed " << mProcessed;
#endif

  // Close calibration if a new run has started or we are receiving data from another run
  if (mInitialized && (tinfo.globalRunNumberChanged == true || mRunNumber != tinfo.runNumber)) {
    if (mProcessed != 0) {
      mProcessed = 0;
      mWorker.endOfRun();
      sendOutput(); // Send output and ask for a reset of the worker
    }
    reset();
    mTimer.Stop();
    LOGF(info, "ZDC Baseline calibration: run change. Timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }

  // Initialization at startup or after a reset
  if (!mInitialized) {
    mInitialized = true;
    updateTimeDependentParams(pc);
    mOutput = &(pc.outputs());
    mHistoFileMetaData = std::make_unique<o2::dataformats::FileMetaData>();
    mHistoFileMetaData->setDataTakingContext(pc.services().get<o2::framework::DataTakingContext>());
    mTimer.Stop();
    mTimer.Reset();
    mTimer.Start(false);
    mRunStartTime = tinfo.creation; // approximate time in ms
    mRunNumber = tinfo.runNumber;   // new current run number
  }

  auto data = pc.inputs().get<o2::zdc::BaselineCalibSummaryData*>("basecalibdata");
  mWorker.process(data.get());
  mProcessed++;
  // Send intermediate calibration data if enough statistics has been collected
  if (mCTimeMod > 0) {
    auto& outd = mWorker.getData();
    auto reft = mCTimeEnd == 0 ? outd.mCTimeBeg : mCTimeEnd;
    if ((reft + mCTimeMod) < outd.mCTimeEnd) {
      // Send output to CCDB but don't reset structures
      mProcessed = 0;
      mWorker.endOfRun();
      sendOutput();
      mCTimeEnd = outd.mCTimeEnd;
    }
  }

  if (mInitialized && pc.transitionState() == TransitionHandlingState::Requested) {
    if (mProcessed != 0) {
      mProcessed = 0;
      mWorker.endOfRun();
      sendOutput(); // Send output and ask for a reset of the worker
    }
    reset();
    mTimer.Stop();
    LOGF(info, "ZDC Baseline calibration: transition. Timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }
}

void BaselineCalibSpec::endOfStream(EndOfStreamContext& ec)
{
  if (mInitialized) {
    if (mProcessed != 0) {
      mProcessed = 0;
      mWorker.endOfRun();
      sendOutput(); // Send output and ask for a reset of the worker
    }
    reset();
    mTimer.Stop();
    LOGF(info, "ZDC Baseline calibration: end of stream. Timing: Cpu: %.3e Real: %.3e s in %d slots", mTimer.CpuTime(), mTimer.RealTime(), mTimer.Counter() - 1);
  }
}

//________________________________________________________________
void BaselineCalibSpec::reset()
{
  mWorker.resetInitFlag();
  mInitialized = false;
}

//________________________________________________________________
void BaselineCalibSpec::sendOutput()
{
  std::string fn = "ZDC_BaselineCalib";

  // extract CCDB infos and calibration objects, convert it to TMemFile and send them to the output
  // TODO in principle, this routine is generic, can be moved to Utils.h
  using clbUtils = o2::calibration::Utils;
  const auto& payload = mWorker.getParamUpd();
  auto& info = mWorker.getCcdbObjectInfo();
  const auto& opt = CalibParamZDC::Instance();
  opt.updateCcdbObjectInfo(info);

  auto image = o2::ccdb::CcdbApi::createObjectImage<BaselineParam>(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  if (mVerbosity > DbgMinimal) {
    payload.print();
  }

  mOutput->snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "ZDCBaselinecalib", 0}, *image.get()); // vector<char>
  mOutput->snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDCBaselinecalib", 0}, info);         // root-serialized

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

framework::DataProcessorSpec getBaselineCalibSpec()
{
  using device = o2::zdc::BaselineCalibSpec;
  using clbUtils = o2::calibration::Utils;

  std::vector<InputSpec> inputs;
  inputs.emplace_back("basecalibdata", "ZDC", "BASECALIBDATA", 0, Lifetime::Sporadic);
  inputs.emplace_back("calibconfig", "ZDC", "BASECALIBCONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathBaselineCalibConfig.data()));
  inputs.emplace_back("moduleconfig", "ZDC", "MODULECONFIG", 0, Lifetime::Condition, o2::framework::ccdbParamSpec(o2::zdc::CCDBPathConfigModule.data()));

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "ZDCBaselinecalib"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "ZDCBaselinecalib"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "zdc-calib-baseline",
    inputs,
    outputs,
    AlgorithmSpec{adaptFromTask<device>()},
    Options{{"verbosity-level", o2::framework::VariantType::Int, 1, {"Verbosity level"}}}};
}

} // namespace zdc
} // namespace o2
