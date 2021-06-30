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
#include "subsysname.h"
#include <array>
#include <gsl/span>
#include <iostream>
#include <unordered_map>
#include <vector>
#include "CommonUtils/ConfigurableParam.h"
#if defined(MUON_SUBSYSTEM_MCH)
#include "MCHConditions/DCSNamer.h"
#define CCDBOBJ "MCH_DPS"
#endif
#if defined(MUON_SUBSYSTEM_MID)
#include "MIDConditions/DCSNamer.h"
#define CCDBOBJ "MID_DPS"
#endif

namespace
{

using DPID = o2::dcs::DataPointIdentifier; // aka alias name
using DPVAL = o2::dcs::DataPointValue;
using DPMAP = std::unordered_map<DPID, std::vector<DPVAL>>;

using namespace o2::calibration;
std::vector<o2::framework::OutputSpec> calibrationOutputs{
  o2::framework::ConcreteDataTypeMatcher{Utils::gDataOriginCDBPayload, CCDBOBJ},
  o2::framework::ConcreteDataTypeMatcher{Utils::gDataOriginCDBWrapper, CCDBOBJ}};

/*
* Create a default CCDB Object Info that will be used as a template.
*
* @param path describes the CCDB data path used (e.g. MCH/LV or MID/HV)
*
* The start and end validity times are supposed to be updated from this template,
* as well as the metadata (if needed). The rest of the information should
* be taken as is.
*/
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

#if defined(MUON_SUBSYSTEM_MCH)
#define NOBJECTS 2
std::array<o2::ccdb::CcdbObjectInfo, NOBJECTS> info{createDefaultInfo("MCH/HV"), createDefaultInfo("MCH/LV")};
#elif defined(MUON_SUBSYSTEM_MID)
#define NOBJECTS 2
std::array<o2::ccdb::CcdbObjectInfo, NOBJECTS> info{createDefaultInfo("MID/HV")};
#endif

std::array<DPMAP, NOBJECTS> dataPoints;

int t0{-1};

/*
* Send a DPMAP to the output.
*
* @param dpmap a map of string to vector of DataPointValue 
* @param output a DPL data allocator
* @param info a CCDB object info describing the dpmap
* @param reason (optional, can be empty) a string description why the dpmap
* was ready to be shipped (e.g. big enough, long enough, end of process, etc...)
*/
void sendOutput(const DPMAP& dpmap, o2::framework::DataAllocator& output, o2::ccdb::CcdbObjectInfo info, const std::string& reason)
{
  if (dpmap.empty()) {
    // we do _not_ write empty objects
    return;
  }
  auto md = info.getMetaData();
  md["upload reason"] = reason;
  info.setMetaData(md);
  auto image = o2::ccdb::CcdbApi::createObjectImage(&dpmap, &info);
  LOG(DEBUG) << "Sending object " << info.getPath() << "/" << info.getFileName() << " of size " << image->size()
             << " bytes, valid for " << info.getStartValidityTimestamp() << " : " << info.getEndValidityTimestamp();
  output.snapshot(o2::framework::Output{Utils::gDataOriginCDBPayload, CCDBOBJ, 0}, *image.get());
  output.snapshot(o2::framework::Output{Utils::gDataOriginCDBWrapper, CCDBOBJ, 0}, info);
}

/*
* Implementation of DPL end of stream callback.
*
* We send the remaining datapoints at the end of the processing.
*/
void endOfStream(o2::framework::EndOfStreamContext& eosc)
{
  LOG(DEBUG) << "This is the end. Must write what we have left ?\n";
  for (auto i = 0; i < NOBJECTS; i++) {
    sendOutput(dataPoints[i], eosc.outputs(), info[i], "end of stream");
  }
}

/*
* Compute the (approximate) size (in KB) of a dpmap.
*/
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

/*
* Compute the duration (in seconds) span by the datapoints in the dpmap.
*/
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

/*
* Decides whether or not the dpmap should be sent to the output.
*
* @param maxSize if the dpmap size is above this size, 
* then it should go to output
* @param maxDuration if the dpmap spans more than this duration, 
* then it should go to output
*
* @returns a boolean stating if the dpmap should be output and a string
* describing why it should be output.
*/
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
  inf.setStartValidityTimestamp(t0);
  //inf.setEndValidityTimestamp(t1);
  return inf;
}

/*
* Process the datapoints received.
*
* The datapoints are accumulated into one (MID) or two (MCH) DPMAPs (map from
* alias names to vector of DataPointValue) : one for HV values (MID and MCH)
* and one for LV values (MCH only).
*
* If the DPMAPs satisfy certain conditions (@see needOutput) they are sent to
* the output.
*
* @param aliases an array of one or two vectors of aliases (one for HV values,
* one for LV values) 
* @param maxSize an array of one or two values for the
* maxsizes of the HV and LV values respectively 
* @param maxDuration an array of
* one or two values for the max durations of the HV and LV values respectively
*/
void processDataPoints(o2::framework::ProcessingContext& pc,
                       std::array<std::vector<std::string>, NOBJECTS> aliases,
                       std::array<int, NOBJECTS> maxSize,
                       std::array<int, NOBJECTS> maxDuration)
{

  auto tfid = o2::header::get<o2::framework::DataProcessingHeader*>(pc.inputs().get("input").header)->startTime;
  if (t0 < 0) {
    t0 = tfid;
  }
  auto dps = pc.inputs().get<gsl::span<o2::dcs::DataPointCompositeObject>>("input");
  for (auto dp : dps) {
    //FIXME: check we're not adding twice the same dp (i.e. check timestamp ?)
    for (auto i = 0; i < NOBJECTS; i++) {
      if (std::find(aliases[i].begin(), aliases[i].end(), dp.id.get_alias()) != aliases[i].end()) {
        dataPoints[i][dp.id].emplace_back(dp.data);
      }
    }
  }
  for (auto i = 0; i < NOBJECTS; i++) {
    auto [shouldOutput, reason] = needOutput(dataPoints[i], maxSize[i], maxDuration[i]);
    if (shouldOutput) {
      auto inf = addTFInfo(info[i], t0, tfid);
      sendOutput(dataPoints[i], pc.outputs(), inf, reason);
      t0 = tfid;
      dataPoints[i].clear(); //FIXME: here the clear should be more clever and keep at least one value per dp ?
    }
  }
}

/*
* Creates the main processing function.
*
* @param ic InitContext which is used to get the options and set the end of 
* stream callback
*/
o2::framework::AlgorithmSpec::ProcessCallback createProcessFunction(o2::framework::InitContext& ic)
{
  auto& callbacks = ic.services().get<o2::framework::CallbackService>();
  callbacks.set(o2::framework::CallbackService::Id::EndOfStream, endOfStream);

  // the aliases arrays contain all the names of the MCH or MID data points
  // we are interested to transit to the CCDB
#if defined(MUON_SUBSYSTEM_MCH)
  std::array<std::vector<std::string>, NOBJECTS> aliases = {
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
#elif defined(MUON_SUBSYSTEM_MID)
  std::array<std::vector<std::string>, NOBJECTS> aliases = {
    o2::mid::dcs::aliases({o2::mid::dcs::MeasurementType::HV_V,
                           o2::mid::dcs::MeasurementType::HV_I})};
  std::array<int, NOBJECTS> maxSize{ic.options().get<int>("hv-max-size")};
  std::array<int, NOBJECTS> maxDuration{ic.options().get<int>("hv-max-duration")};
#endif

  for (auto i = 0; i < NOBJECTS; i++) {
    dataPoints[i].clear();
  }

  return [aliases, maxSize, maxDuration](o2::framework::ProcessingContext& pc) {
    processDataPoints(pc, aliases, maxSize, maxDuration);
  };
}

/* Helper function to create a ConfigParamSpec option object.
* 
* @param name is either 'size' or 'duration'
* @param value is the default value to be used (i.e. when the option is not 
* specified on the command line)
* @param what is either 'hv' or 'lv'
* @param unit is the unit in which the values are given
*/
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

/**
* DPL Workflow to process MCH or MID DCS data points.
*
* The expected input is a vector of DataPointCompositeObject containing
* only MCH (or only MID) data points.
*
* Those datapoints are accumulated into DPMAPs (map from alias names to
* vector of DataPointValue).
*
* The accumulated DPMAPs are sent to the output whenever :
* - they reach a given size (--xx-max-size option(s))
* - they span a given duration (--xx-max-duration option(s))
* - the workflow is ended
*
*/
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DataProcessorSpec dcsProcessor;

  AlgorithmSpec algo(createProcessFunction);

  dcsProcessor.name = fmt::format("{}-dcs-processor", o2::muon::subsysname());
  dcsProcessor.inputs = o2::framework::Inputs
  {
#if defined(MUON_SUBSYSTEM_MCH)
    {
      "input", "DCS", "MCHDATAPOINTS"
    }
  };
#elif defined(MUON_SUBSYSTEM_MID)
    {
      "input", "DCS", "MIDDATAPOINTS"
    }
  };
#endif
  dcsProcessor.outputs = calibrationOutputs;
  dcsProcessor.algorithm = algo;
  dcsProcessor.options = {
#if defined(MUON_SUBSYSTEM_MCH)
    whenToSendOption("lv", 128, "size", "KB"),
    whenToSendOption("lv", 8 * 3600, "duration", "seconds"),
#endif
    whenToSendOption("hv", 128, "size", "KB"),
    whenToSendOption("hv", 8 * 3600, "duration", "seconds")};

  return {dcsProcessor};
}
