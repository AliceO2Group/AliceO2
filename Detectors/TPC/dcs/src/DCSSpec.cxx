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

/// @file   DCSSpec.cxx
/// @author Jens Wiechula
/// @brief  DCS processing

#include <chrono>
#include <vector>
#include <string>
#include <string_view>
#include <unordered_map>
#include <TStopwatch.h>
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"
#include "Framework/Logger.h"
#include "Framework/ConfigParamRegistry.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsDCS/DataPointIdentifier.h"
#include "DetectorsDCS/DataPointValue.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "DetectorsDCS/DeliveryType.h"
#include "DetectorsDCS/AliasExpander.h"

#include "TPCBaseRecSim/CDBInterface.h"
#include "TPCdcs/DCSProcessor.h"
#include "TPCdcs/DCSSpec.h"

using namespace o2::framework;
using DPCOM = o2::dcs::DataPointCompositeObject;
using HighResClock = std::chrono::high_resolution_clock;
using Duration = std::chrono::duration<double, std::ratio<1, 1>>;
constexpr auto CDBPayload = o2::calibration::Utils::gDataOriginCDBPayload;
constexpr auto CDBWrapper = o2::calibration::Utils::gDataOriginCDBWrapper;

namespace o2::tpc
{

const std::unordered_map<CDBType, o2::header::DataDescription> CDBDescMap{
  {CDBType::CalTemperature, o2::header::DataDescription{"TPC_Temperature"}},
  {CDBType::CalHV, o2::header::DataDescription{"TPC_HighVoltage"}},
  {CDBType::CalGas, o2::header::DataDescription{"TPC_Gas"}},
  {CDBType::CalPressure, o2::header::DataDescription{"TPC_Pressure"}}};

class DCSDevice : public o2::framework::Task
{
 public:
  DCSDevice() = default;

  void init(o2::framework::InitContext& ic) final;

  void run(o2::framework::ProcessingContext& pc) final;

  template <typename T>
  void sendObject(DataAllocator& output, T& obj, const CDBType calibType);

  void updateCCDB(DataAllocator& output);

  void finalizeDCS(DataAllocator& output)
  {
    if (mDCS.hasData()) {
      mDCS.finalizeSlot();
      if (mWriteDebug) {
        mDCS.writeDebug();
      }
      updateCCDB(output);
      mDCS.reset();
    }
  }

  void finalize()
  {
    mDCS.finalize();
  }

  void endOfStream(o2::framework::EndOfStreamContext& ec) final
  {
    LOGP(info, "endOfStream");
    finalizeDCS(ec.outputs()); // TODO: can this also be done in stop() due to CCDB?
    finalize();
  }

  void stop() final
  {
    LOGP(info, "stop");
    finalize();
  }

 private:
  DCSProcessor mDCS;
  CDBStorage mCDBStorage;
  HighResClock::time_point mIntervalStartTime;
  uint64_t mUpdateIntervalStart{0};
  uint64_t mLastCreationTime{0};
  int mCCDBupdateInterval;
  int mFitInterval;
  bool mDebugWritten{false};
  bool mWriteDebug{false};
  bool mReportTiming{false};
  int mUpdateIntervalnTFs{-1};
};

void DCSDevice::init(o2::framework::InitContext& ic)
{
  mWriteDebug = ic.options().get<bool>("write-debug");
  mCCDBupdateInterval = ic.options().get<int>("update-interval");
  mFitInterval = ic.options().get<int>("fit-interval");
  const int pressureInterval = ic.options().get<int>("pressure-interval");
  const int pressureIntervalRef = ic.options().get<int>("pressure-ref-interval");
  if (mCCDBupdateInterval < 0) {
    mCCDBupdateInterval = 0;
  }
  if (mFitInterval >= mCCDBupdateInterval) {
    LOGP(info, "fit interval {} >= ccdb update interval {}, making them identical", mFitInterval, mCCDBupdateInterval);
    mFitInterval = mCCDBupdateInterval;
  }
  mUpdateIntervalnTFs = ic.options().get<int>("update-interval-nTFs");

  mDCS.setFitInterval(mFitInterval * 1000); // in ms in mDCS
  mDCS.setPressureInterval(pressureInterval * 1000);
  mDCS.setRefPressureInterval(pressureIntervalRef * 1000);
  mDCS.setRoundToInterval(ic.options().get<bool>("round-to-interval"));

  // set default meta data
  mCDBStorage.setResponsible("Jens Wiechula (jens.wiechula@cern.ch)");
  mCDBStorage.setIntervention(CDBIntervention::Automatic);
  mCDBStorage.setReason("DCS workflow upload");
  mReportTiming = ic.options().get<bool>("report-timing");
}

void DCSDevice::run(o2::framework::ProcessingContext& pc)
{
  TStopwatch sw;
  mLastCreationTime = pc.services().get<o2::framework::TimingInfo>().creation;
  if (mUpdateIntervalStart == 0) {
    mUpdateIntervalStart = mLastCreationTime;
  }
  if (mUpdateIntervalnTFs > 0 && (pc.services().get<o2::framework::TimingInfo>().tfCounter % mUpdateIntervalnTFs == 0)) {
    // finalize DCS for every n-TFs (useful for testing purpose when reading in data from local file)
    finalizeDCS(pc.outputs());
  } else {
    if (mLastCreationTime - mUpdateIntervalStart >= uint64_t(mCCDBupdateInterval * 1000)) {
      finalizeDCS(pc.outputs());
      mUpdateIntervalStart = mLastCreationTime;
    }
  }

  auto dps = pc.inputs().get<gsl::span<DPCOM>>("input");
  mDCS.process(dps);
  sw.Stop();
  if (mReportTiming) {
    LOGP(info, "Timing CPU:{:.3e} Real:{:.3e} at slice {}", sw.CpuTime(), sw.RealTime(), pc.services().get<o2::framework::TimingInfo>().timeslice);
  }
}

template <typename T>
void DCSDevice::sendObject(DataAllocator& output, T& obj, const CDBType calibType)
{
  LOGP(info, "Prepare CCDB for {}", CDBTypeMap.at(calibType));

  std::map<std::string, std::string> md = mCDBStorage.getMetaData();
  o2::ccdb::CcdbObjectInfo w;
  // for online processing extend the validity range. Will be truncated with the adjustableEOV procedure
  o2::calibration::Utils::prepareCCDBobjectInfo(obj, w, CDBTypeMap.at(calibType), md, mUpdateIntervalStart, mLastCreationTime + 2 * mCCDBupdateInterval * 1000);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&obj, &w);

  LOGP(info, "Sending object {} / {} of size {} bytes, valid for {} : {} ", w.getPath(), w.getFileName(), image->size(), w.getStartValidityTimestamp(), w.getEndValidityTimestamp());
  output.snapshot(Output{CDBPayload, CDBDescMap.at(calibType), 0}, *image.get());
  output.snapshot(Output{CDBWrapper, CDBDescMap.at(calibType), 0}, w);
}

void DCSDevice::updateCCDB(DataAllocator& output)
{
  sendObject(output, mDCS.getTemperature(), CDBType::CalTemperature);
  sendObject(output, mDCS.getHighVoltage(), CDBType::CalHV);
  sendObject(output, mDCS.getGas(), CDBType::CalGas);
  sendObject(output, mDCS.getPressure(), CDBType::CalPressure);
}

/// ===| create DCS processor |=================================================
///
///
DataProcessorSpec getDCSSpec()
{

  std::vector<OutputSpec> outputs;
  outputs.emplace_back(ConcreteDataTypeMatcher{CDBPayload, CDBDescMap.at(CDBType::CalTemperature)}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{CDBWrapper, CDBDescMap.at(CDBType::CalTemperature)}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{CDBPayload, CDBDescMap.at(CDBType::CalHV)}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{CDBWrapper, CDBDescMap.at(CDBType::CalHV)}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{CDBPayload, CDBDescMap.at(CDBType::CalGas)}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{CDBWrapper, CDBDescMap.at(CDBType::CalGas)}, Lifetime::Sporadic);

  outputs.emplace_back(ConcreteDataTypeMatcher{CDBPayload, CDBDescMap.at(CDBType::CalPressure)}, Lifetime::Sporadic);
  outputs.emplace_back(ConcreteDataTypeMatcher{CDBWrapper, CDBDescMap.at(CDBType::CalPressure)}, Lifetime::Sporadic);

  return DataProcessorSpec{
    "tpc-dcs",
    Inputs{{"input", "DCS", "TPCDATAPOINTS"}},
    outputs,
    AlgorithmSpec{adaptFromTask<DCSDevice>()},
    Options{
      {"write-debug", VariantType::Bool, false, {"write a debug output tree"}},
      {"report-timing", VariantType::Bool, false, {"Report timing for every slice"}},
      {"update-interval", VariantType::Int, 60 * 15, {"update interval in seconds for which ccdb entries are written"}},
      {"fit-interval", VariantType::Int, 60 * 5, {"interval in seconds for which to e.g. perform fits of the temperature sensors"}},
      {"round-to-interval", VariantType::Bool, false, {"round fit interval to fixed times e.g. to every 5min in the hour"}},
      {"pressure-interval", VariantType::Int, 100, {"interval in seconds for which to average the pressure values"}},
      {"pressure-ref-interval", VariantType::Int, 24 * 60 * 60, {"interval in seconds for which to calculate the reference pressure values"}},
      {"update-interval-nTFs", VariantType::Int, -1, {"only used when larger than 0: update interval in nTFs for which ccdb entries are written "}}} // end Options
  }; // end DataProcessorSpec
}

} // end namespace o2::tpc
