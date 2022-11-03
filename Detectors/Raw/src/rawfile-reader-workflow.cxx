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

#include "RawFileReaderWorkflow.h"
#include "DetectorsRaw/RawFileReader.h"
#include "CommonUtils/NameConf.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ChannelSpecHelpers.h"
#include "Framework/Logger.h"
#include <string>
#include <bitset>

using namespace o2::framework;
using namespace o2::raw;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"input-conf", VariantType::String, "", {"configuration file with input (obligatory)"}});
  options.push_back(ConfigParamSpec{"min-tf", VariantType::Int64, 0L, {"min TF ID to process"}});
  options.push_back(ConfigParamSpec{"max-tf", VariantType::Int64, 0xffffffffL, {"max TF ID to process"}});
  options.push_back(ConfigParamSpec{"run-number", VariantType::Int, 0, {"impose run number"}});
  options.push_back(ConfigParamSpec{"loop", VariantType::Int, 1, {"loop N times (infinite for N<0)"}});
  options.push_back(ConfigParamSpec{"delay", VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"buffer-size", VariantType::Int64, 5 * 1024L, {"buffer size for files preprocessing"}});
  options.push_back(ConfigParamSpec{"super-page-size", VariantType::Int64, 1024L * 1024L, {"super-page size for FMQ parts definition"}});
  options.push_back(ConfigParamSpec{"part-per-sp", VariantType::Bool, false, {"FMQ parts per superpage instead of per HBF"}});
  options.push_back(ConfigParamSpec{"raw-channel-config", VariantType::String, "", {"optional raw FMQ channel for non-DPL output"}});
  options.push_back(ConfigParamSpec{"cache-data", VariantType::Bool, false, {"cache data at 1st reading, may require excessive memory!!!"}});
  options.push_back(ConfigParamSpec{"detect-tf0", VariantType::Bool, false, {"autodetect HBFUtils start Orbit/BC from 1st TF seen"}});
  options.push_back(ConfigParamSpec{"calculate-tf-start", VariantType::Bool, false, {"calculate TF start instead of using TType"}});
  options.push_back(ConfigParamSpec{"drop-tf", VariantType::String, "none", {"Drop each TFid%(1)==(2) of detector, e.g. ITS,2,4;TPC,4[,0];..."}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});
  options.push_back(ConfigParamSpec{"send-diststf-0xccdb", VariantType::Bool, false, {"send explicit FLP/DISTSUBTIMEFRAME/0xccdb output"}});
  options.push_back(ConfigParamSpec{"hbfutils-config", VariantType::String, std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE), {"configKeyValues ini file for HBFUtils (used if exists)"}});
  options.push_back(ConfigParamSpec{"timeframes-shm-limit", VariantType::String, "0", {"Minimum amount of SHM required in order to publish data"}});
  options.push_back(ConfigParamSpec{"metric-feedback-channel-format", VariantType::String, "name=metric-feedback,type=pull,method=connect,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0", {"format for the metric-feedback channel for TF rate limiting"}});
  // options for error-check suppression

  for (int i = 0; i < RawFileReader::NErrorsDefined; i++) {
    auto ei = RawFileReader::ErrTypes(i);
    options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(ei), VariantType::Bool, false, {RawFileReader::nochk_expl(ei)}});
  }
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  ReaderInp rinp;
  rinp.inifile = configcontext.options().get<std::string>("input-conf");
  rinp.loop = configcontext.options().get<int>("loop");
  rinp.maxTF = uint32_t(configcontext.options().get<int64_t>("max-tf"));
  rinp.minTF = uint32_t(configcontext.options().get<int64_t>("min-tf"));
  rinp.runNumber = configcontext.options().get<int>("run-number");
  rinp.bufferSize = uint64_t(configcontext.options().get<int64_t>("buffer-size"));
  rinp.spSize = uint64_t(configcontext.options().get<int64_t>("super-page-size"));
  rinp.partPerSP = configcontext.options().get<bool>("part-per-sp");
  rinp.cache = configcontext.options().get<bool>("cache-data");
  rinp.autodetectTF0 = configcontext.options().get<bool>("detect-tf0");
  rinp.preferCalcTF = configcontext.options().get<bool>("calculate-tf-start");
  rinp.rawChannelConfig = configcontext.options().get<std::string>("raw-channel-config");
  rinp.delay_us = uint32_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds
  rinp.dropTF = configcontext.options().get<std::string>("drop-tf");
  rinp.sup0xccdb = !configcontext.options().get<bool>("send-diststf-0xccdb");
  rinp.errMap = 0;
  for (int i = RawFileReader::NErrorsDefined; i--;) {
    auto ei = RawFileReader::ErrTypes(i);
    bool defOpt = RawFileReader::ErrCheckDefaults[i];
    if (configcontext.options().get<bool>(RawFileReader::nochk_opt(ei).c_str()) ? !defOpt : defOpt) { // cmdl option inverts default!
      rinp.errMap |= 0x1 << i;
    }
  }
  rinp.minSHM = std::stoul(configcontext.options().get<std::string>("timeframes-shm-limit"));
  int rateLimitingIPCID = std::stoi(configcontext.options().get<std::string>("timeframes-rate-limit-ipcid"));
  std::string chanFmt = configcontext.options().get<std::string>("metric-feedback-channel-format");
  if (rateLimitingIPCID > -1 && !chanFmt.empty()) {
    rinp.metricChannel = fmt::format(chanFmt, o2::framework::ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
  }
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  auto hbfini = configcontext.options().get<std::string>("hbfutils-config");
  if (!hbfini.empty() && o2::conf::ConfigurableParam::configFileExists(hbfini)) {
    o2::conf::ConfigurableParam::updateFromFile(hbfini, "HBFUtils", true); // update only those values which were not touched yet (provenance == kCODE)
  }
  return std::move(o2::raw::getRawFileReaderWorkflow(rinp));
}
