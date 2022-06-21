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

/// @file GRPDCSDPsSpec.cxx

#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include "GRPWorkflows/GRPDCSDPsSpec.h"
#include "GRPCalibration/GRPDCSDPsProcessor.h"
#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "CommonTypes/Units.h"
#include "DetectorsDCS/AliasExpander.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include <TStopwatch.h>

namespace o2
{
namespace grp
{

using CcdbManager = o2::ccdb::BasicCCDBManager;
using namespace o2::ccdb;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPCOM = o2::dcs::DataPointCompositeObject;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;

void GRPDCSDPsDataProcessor::init(o2::framework::InitContext& ic)
{

  std::vector<DPID> vect;
  mDPsUpdateInterval = ic.options().get<int64_t>("DPs-update-interval");
  if (mDPsUpdateInterval == 0) {
    LOG(error) << "GRP DPs update interval set to zero seconds --> changed to 60";
    mDPsUpdateInterval = 60;
  }
  bool useCCDBtoConfigure = ic.options().get<bool>("use-ccdb-to-configure");
  if (useCCDBtoConfigure) {
    LOG(info) << "Configuring via CCDB";
    std::string ccdbpath = ic.options().get<std::string>("ccdb-path");
    auto& mgr = CcdbManager::instance();
    mgr.setURL(ccdbpath);
    CcdbApi api;
    api.init(mgr.getURL());
    long ts = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count();
    std::unordered_map<DPID, std::string>* dpid2DataDesc = mgr.getForTimeStamp<std::unordered_map<DPID, std::string>>("GRP/Config/DCSDPconfig", ts);
    for (auto& i : *dpid2DataDesc) {
      vect.push_back(i.first);
    }
  } else {
    LOG(info) << "Configuring via hardcoded strings";
    std::vector<std::string> aliasesBFieldDouble = {"L3Current", "DipoleCurrent"};
    std::vector<std::string> aliasesBFieldBool = {"L3Polarity", "DipolePolarity"};
    std::vector<std::string> aliasesEnvVar = {"CavernTemperature", "CavernAtmosPressure", "SurfaceAtmosPressure", "CavernAtmosPressure2"};
    std::vector<std::string> compactAliasesLHCIFDouble = {"LHC_IntensityBeam[1..2]_totalIntensity", "ALI_Background[1..3]",
                                                          "ALI_Lumi_Total_Inst",
                                                          "BPTX_deltaT_B1_B2", "BPTX_deltaTRMS_B1_B2",
                                                          "BPTX_Phase_B[1..2]", "BPTX_PhaseRMS_B[1..2]", "BPTX_Phase_Shift_B[1..2]"};
    std::vector<std::string> aliasesLHCIFDouble = o2::dcs::expandAliases(compactAliasesLHCIFDouble);
    std::vector<std::string> aliasesLHCIFString = {"ALI_Lumi_Source_Name", "MACHINE_MODE", "BEAM_MODE"};
    std::vector<std::string> aliasesLHCIFCollimators = {"LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_gap_upstream",
                                                        "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_left_upstream",
                                                        "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_downstream", "LHC_CollimatorPos_TCLIA_4R2_lvdt_right_upstream"};

    for (const auto& i : aliasesBFieldDouble) {
      vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
    }
    for (const auto& i : aliasesBFieldBool) {
      vect.emplace_back(i, o2::dcs::DPVAL_BOOL);
    }
    for (const auto& i : aliasesEnvVar) {
      vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
    }
    for (const auto& i : aliasesLHCIFDouble) {
      vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
    }
    for (const auto& i : aliasesLHCIFCollimators) {
      vect.emplace_back(i, o2::dcs::DPVAL_DOUBLE);
    }
    for (const auto& i : aliasesLHCIFString) {
      vect.emplace_back(i, o2::dcs::DPVAL_STRING);
    }
  }

  LOG(info) << "Listing Data Points for GRP:";
  for (auto& i : vect) {
    LOG(info) << i;
  }

  mProcessor = std::make_unique<o2::grp::GRPDCSDPsProcessor>();
  mVerbose = ic.options().get<bool>("use-verbose-mode");
  LOG(info) << " ************************* Verbose?" << mVerbose;
  bool clearVectors = ic.options().get<bool>("clear-vectors");
  LOG(info) << " ************************* Clear vectors?" << clearVectors;
  if (mVerbose) {
    mProcessor->useVerboseMode();
  }
  if (clearVectors) {
    mProcessor->clearVectors();
  }
  mProcessor->init(vect);
  mTimer = HighResClock::now();
  mReportTiming = ic.options().get<bool>("report-timing") || mVerbose;
}
//__________________________________________________________________

void GRPDCSDPsDataProcessor::run(o2::framework::ProcessingContext& pc)
{
  mLHCIFupdated = false;
  TStopwatch sw;
  auto startValidity = DataRefUtils::getHeader<DataProcessingHeader*>(pc.inputs().getFirstValid(true))->creation;
  auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
  auto timeNow = HighResClock::now();
  if (startValidity == 0xffffffffffffffff) {                                                                   // it means it is not set
    startValidity = std::chrono::duration_cast<std::chrono::milliseconds>(timeNow.time_since_epoch()).count(); // in ms
  }
  mProcessor->setStartValidity(startValidity);
  mProcessor->process(dps);
  Duration elapsedTime = timeNow - mTimer; // in seconds
  if (elapsedTime.count() >= mDPsUpdateInterval || mProcessor->isLHCIFInfoUpdated()) {
    // after enough time or after something changed, we store the LHCIF part of the DPs:
    if (elapsedTime.count() >= mDPsUpdateInterval) {
      if (mVerbose) {
        LOG(info) << "enough time passed (" << elapsedTime.count() << " s), sending to CCDB LHCIFDPs";
      }
    } else {
      if (mVerbose) {
        LOG(info) << "sending to CCDB LHCIFDPs since something changed";
      }
    }
    mProcessor->updateLHCIFInfoCCDB();
    sendLHCIFDPsoutput(pc.outputs());
    mProcessor->resetAndKeepLastLHCIFDPs();
    mLHCIFupdated = true;
    mProcessor->resetPIDsLHCIF();
  }
  if (elapsedTime.count() >= mDPsUpdateInterval) {
    // after enough time, we store:
    // collimators:
    if (mVerbose) {
      LOG(info) << "enough time passed (" << elapsedTime.count() << " s), sending to CCDB Env and Coll";
    }
    mProcessor->updateCollimatorsCCDB();
    sendCollimatorsDPsoutput(pc.outputs());
    mProcessor->resetAndKeepLast(mProcessor->getCollimatorsObj().mCollimators);
    // env vars:
    mProcessor->updateEnvVarsCCDB();
    sendEnvVarsDPsoutput(pc.outputs());
    mProcessor->resetAndKeepLast(mProcessor->getEnvVarsObj().mEnvVars);
    mTimer = timeNow;
    mProcessor->resetPIDs();
  }
  if (mProcessor->isMagFieldUpdated()) {
    sendMagFieldDPsoutput(pc.outputs());
  }
  sw.Stop();
  if (mReportTiming) {
    LOGP(info, "Timing CPU:{:.3e} Real:{:.3e} at slice {}", sw.CpuTime(), sw.RealTime(), pc.services().get<o2::framework::TimingInfo>().timeslice);
  }
}
//________________________________________________________________

void GRPDCSDPsDataProcessor::endOfStream(o2::framework::EndOfStreamContext& ec)
{

  LOG(info) << " ********** End of Stream **********";
  // we force writing to CCDB the entries for which we accumulate values in vectors (we don't do it for the B field
  // because this is updated every time on change of any of the 4 DPs related to it)
  if (!mLHCIFupdated) { // the last TF did not update the LHCIF CCDB entry, let's force it
    mProcessor->updateLHCIFInfoCCDB();
    sendLHCIFDPsoutput(ec.outputs());
  }
  mProcessor->updateCollimatorsCCDB();
  sendCollimatorsDPsoutput(ec.outputs());

  mProcessor->updateEnvVarsCCDB();
  sendEnvVarsDPsoutput(ec.outputs());
}

//________________________________________________________________

void GRPDCSDPsDataProcessor::sendLHCIFDPsoutput(DataAllocator& output)
{
  // filling CCDB with LHCIF DPs object

  const auto& payload = mProcessor->getLHCIFObj();
  auto& info = mProcessor->getccdbLHCIFInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_LHCIF_DPs", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_LHCIF_DPs", 0}, info);
}
//________________________________________________________________

void GRPDCSDPsDataProcessor::sendMagFieldDPsoutput(DataAllocator& output)
{
  // filling CCDB with B field object

  const auto& payload = mProcessor->getMagFieldObj();
  auto& info = mProcessor->getccdbMagFieldInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_Bfield", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_Bfield", 0}, info);
}

//________________________________________________________________

void GRPDCSDPsDataProcessor::sendCollimatorsDPsoutput(DataAllocator& output)
{
  // filling CCDB with Collimators object

  const auto& payload = mProcessor->getCollimatorsObj();
  auto& info = mProcessor->getccdbCollimatorsInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_COLLIM_DPs", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_COLLIM_DPs", 0}, info);
}

//________________________________________________________________

void GRPDCSDPsDataProcessor::sendEnvVarsDPsoutput(DataAllocator& output)
{
  // filling CCDB with EnvVars object

  const auto& payload = mProcessor->getEnvVarsObj();
  auto& info = mProcessor->getccdbEnvVarsInfo();
  auto image = o2::ccdb::CcdbApi::createObjectImage(&payload, &info);
  LOG(info) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_EVARS_DPs", 0}, *image.get());
  output.snapshot(Output{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_EVARS_DPs", 0}, info);
}

} // namespace grp

namespace framework
{

DataProcessorSpec getGRPDCSDPsDataProcessorSpec()
{
  using clbUtils = o2::calibration::Utils;

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_Bfield"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_Bfield"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_LHCIF_DPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_LHCIF_DPs"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_COLLIM_DPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_COLLIM_DPs"}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBPayload, "GRP_EVARS_DPs"}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{o2::calibration::Utils::gDataOriginCDBWrapper, "GRP_EVARS_DPs"}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "grp-dcs-data-processor",
    Inputs{{"input", "DCS", "GRPDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<o2::grp::GRPDCSDPsDataProcessor>()},
    Options{{"ccdb-path", VariantType::String, "http://localhost:8080", {"Path to CCDB"}},
            {"use-ccdb-to-configure", VariantType::Bool, false, {"Use CCDB to configure"}},
            {"use-verbose-mode", VariantType::Bool, false, {"Use verbose mode"}},
            {"report-timing", VariantType::Bool, false, {"Report timing for every slice"}},
            {"DPs-update-interval", VariantType::Int64, 600ll, {"Interval (in s) after which to update the DPs CCDB entry"}},
            {"clear-vectors", VariantType::Bool, false, {"Clear vectors when starting processing for a new CCDB entry (latest value will not be kept)"}}}};
}

} // namespace framework
} // namespace o2
