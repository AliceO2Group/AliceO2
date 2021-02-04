// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "CCDB/CcdbApi.h"
#include "CCDB/CcdbObjectInfo.h"
#include "CommonUtils/MemFileHelper.h"
#include "DetectorsCalibration/Utils.h"
#include "DetectorsDCS/DataPointCompositeObject.h"
#include "Framework/CallbackService.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/runDataProcessing.h"
#include "MCHConditions/DCSNamer.h"
#include <gsl/span>
#include <iostream>
#include <unordered_map>
#include <array>
#include <vector>

namespace
{

using namespace o2::calibration;
using DPID = o2::dcs::DataPointIdentifier;
using DPVAL = o2::dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

std::vector<o2::framework::OutputSpec> calibrationOutputs{
  o2::framework::ConcreteDataTypeMatcher{Utils::gDataOriginCLB, Utils::gDataDescriptionCLBPayload},
  o2::framework::ConcreteDataTypeMatcher{Utils::gDataOriginCLB, Utils::gDataDescriptionCLBInfo}};

std::array<DPMAP, 2> dataPoints;
int t0{-1};

o2::ccdb::CcdbObjectInfo createDefaultInfo(const char* path)
{
  DPMAP obj;
  auto clName = o2::utils::MemFileHelper::getClassName(obj);
  auto flName = o2::ccdb::CcdbApi::generateFileName(clName);
  o2::ccdb::CcdbObjectInfo info;
  info.setPath(path);
  info.setObjectType(clName);
  info.setFileName(flName);
  info.setStartValidityTimestamp(0);
  info.setEndValidityTimestamp(99999999999999);
  std::map<std::string, std::string> md;
  info.setMetaData(md);
  return info;
}

std::array<o2::ccdb::CcdbObjectInfo, 2> info{createDefaultInfo("MCH/HV"), createDefaultInfo("MCH/LV")};

void sendOutput(const DPMAP& dpmap, o2::framework::DataAllocator& output, o2::ccdb::CcdbObjectInfo info, const std::string& reason)
{
  if (dpmap.empty()) {
    // do not write empty objects
    return;
  }
  auto md = info.getMetaData();
  md["upload reason"] = reason;
  info.setMetaData(md);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&dpmap, &info);
  LOG(INFO) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
            << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(o2::framework::Output{Utils::gDataOriginCLB, Utils::gDataDescriptionCLBPayload, 0}, *image.get());
  output.snapshot(o2::framework::Output{Utils::gDataOriginCLB, Utils::gDataDescriptionCLBInfo, 0}, info);
}

void endOfStream(o2::framework::EndOfStreamContext& eosc)
{
  std::cout << "This is the end. Must write what we have left ?\n";
  for (auto i = 0; i < 2; i++) {
    sendOutput(dataPoints[i], eosc.outputs(), info[i], "end of stream");
  }
}

size_t computeSize(const DPMAP& dpmap)
{
  constexpr int itemSize = 64; // DataPointIdentifier or DataPointValue have the same size = 64 bytes
  constexpr float byte2KB = 1.0 / 1024;

  size_t nofItems = 0;
  for (auto did : dpmap) {
    nofItems++;                    // +1 for the DataPointIdentifier itself
    nofItems += did.second.size(); // +number of DataPointValues
  }
  return static_cast<size_t>(std::floor(nofItems * itemSize * byte2KB));
}

int computeDuration(const DPMAP& dpmap)
{
  uint64_t minTime{std::numeric_limits<uint64_t>::max()};
  uint64_t maxTime{0};

  for (auto did : dpmap) {
    for (auto d : did.second) {
      minTime = std::min(minTime, d.get_epoch_time());
      maxTime = std::max(maxTime, d.get_epoch_time());
    }
  }
  return static_cast<int>((maxTime - minTime) / 1000);
}

std::tuple<bool, std::string> needOutput(const DPMAP& dpmap, int maxSize, int maxDuration)
{
  std::string reason;

  if (dpmap.empty()) {
    return {false, reason};
  }

  bool bigEnough{false};
  bool longEnough{false};
  bool complete{true}; // FIXME: should check here that we indeed have all our dataPoints

  if (maxSize && (computeSize(dpmap) > maxSize)) {
    bigEnough = true;
    reason += "[big enough]";
  }

  if (maxDuration) {
    auto seconds = computeDuration(dpmap);
    if (seconds > maxDuration) {
      longEnough = true;
      reason += fmt::format("[long enough ({} s)]", seconds);
    }
  }

  return {complete && (bigEnough || longEnough), reason};
}

o2::ccdb::CcdbObjectInfo addTFInfo(o2::ccdb::CcdbObjectInfo inf,
                                   uint64_t t0, uint64_t t1)
{
  auto md = inf.getMetaData();
  md["tf range"] = fmt::format("{}-{}", t0, t1);
  inf.setMetaData(md);
  return inf;
}

void processDataPoints(o2::framework::ProcessingContext& pc,
                       std::array<std::vector<std::string>, 2> aliases,
                       std::array<int, 2> maxSize,
                       std::array<int, 2> maxDuration)
{

  auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
  if (t0 < 0) {
    t0 = tfid;
  }
  auto dps = pc.inputs().get<gsl::span<o2::dcs::DataPointCompositeObject>>("input");
  for (auto dp : dps) {
    //FIXME: check we're not adding twice the same dp (i.e. check timestamp ?)
    for (auto i = 0; i < 2; i++) {
      if (std::find(aliases[i].begin(), aliases[i].end(), dp.id.get_alias()) != aliases[i].end()) {
        dataPoints[i][dp.id].emplace_back(dp.data);
      }
    }
  }
  for (auto i = 0; i < 2; i++) {
    auto [shouldOutput, reason] = needOutput(dataPoints[i], maxSize[i], maxDuration[i]);
    if (shouldOutput) {
      auto inf = addTFInfo(info[i], t0, tfid);
      sendOutput(dataPoints[i], pc.outputs(), inf, reason);
      t0 = tfid;
      dataPoints[i].clear(); //FIXME: here the clear should be more clever and keep at least one value per dp
    }
  }
}

o2::framework::AlgorithmSpec::ProcessCallback createProcessFunction(o2::framework::InitContext& ic)
{
  auto& callbacks = ic.services().get<o2::framework::CallbackService>();
  callbacks.set(o2::framework::CallbackService::Id::EndOfStream, endOfStream);

  std::array<std::vector<std::string>, 2> aliases = {
    o2::mch::dcs::aliases({o2::mch::dcs::MeasurementType::HV_V,
                           o2::mch::dcs::MeasurementType::HV_I}),
    o2::mch::dcs::aliases({o2::mch::dcs::MeasurementType::LV_V_FEE_ANALOG,
                           o2::mch::dcs::MeasurementType::LV_V_FEE_DIGITAL,
                           o2::mch::dcs::MeasurementType::LV_V_SOLAR})};

  std::array<int, 2> maxSize{
    ic.options().get<int>("hv-max-size"),
    ic.options().get<int>("lv-max-size")};

  std::array<int, 2> maxDuration{
    ic.options().get<int>("hv-max-duration"),
    ic.options().get<int>("lv-max-duration")};

  for (auto i = 0; i < 2; i++) {
    dataPoints[i].clear();
  }

  return [aliases, maxSize, maxDuration](o2::framework::ProcessingContext& pc) {
    processDataPoints(pc, aliases, maxSize, maxDuration);
  };
}

o2::framework::ConfigParamSpec whenToSendOption(const char* name, int value,
                                                const char* what, const char* unit)
{
  std::string uname(name);
  o2::dcs::to_upper_case(uname);

  std::string description = fmt::format(R"(max {} calibration object {} (in {}).
When that {} is reached the object is shipped. Use 0 to disable this check.)",
                                        uname, what, unit, what);

  return {fmt::format("{}-max-{}", name, what),
          o2::framework::VariantType::Int,
          value,
          {description}};
}

} // namespace

using o2::framework::AlgorithmSpec;
using o2::framework::ConfigContext;
using o2::framework::DataProcessorSpec;
using o2::framework::WorkflowSpec;

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DataProcessorSpec dcsProcessor;

  AlgorithmSpec algo(createProcessFunction);

  dcsProcessor.name = "mch-dcs-processor";
  dcsProcessor.inputs = o2::framework::Inputs{{"input", "DCS", "MCHDATAPOINTS"}};
  dcsProcessor.outputs = calibrationOutputs;
  dcsProcessor.algorithm = algo;
  dcsProcessor.options = {
    whenToSendOption("hv", 128, "size", "KB"),
    whenToSendOption("lv", 128, "size", "KB"),
    whenToSendOption("hv", 8 * 3600, "duration", "seconds"),
    whenToSendOption("lv", 8 * 3600, "duration", "seconds")};

  return {dcsProcessor};
}
