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

#include "CommonUtils/ConfigurableParam.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/Logger.h"
#include <string>
#include <bitset>
#include "TFReaderSpec.h"

using namespace o2::framework;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"input-data", VariantType::String, "", {"input data (obligatory)"}});
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, "all", {"list of dectors"}});
  options.push_back(ConfigParamSpec{"raw-only-det", VariantType::String, "none", {"do not open non-raw channel for these detectors"}});
  options.push_back(ConfigParamSpec{"non-raw-only-det", VariantType::String, "none", {"do not open raw channel for these detectors"}});
  options.push_back(ConfigParamSpec{"max-tf", VariantType::Int, -1, {"max TF ID to process (<= 0 : infinite)"}});
  options.push_back(ConfigParamSpec{"loop", VariantType::Int, 0, {"loop N times (-1 = infinite)"}});
  options.push_back(ConfigParamSpec{"delay", VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"copy-cmd", VariantType::String, "alien_cp ?src file://?dst", {"copy command for remote files"}}); // Use "XrdSecPROTOCOL=sss,unix xrdcp -N root://eosaliceo2.cern.ch/?src ?dst" for direct EOS access
  options.push_back(ConfigParamSpec{"tf-file-regex", VariantType::String, ".+\\.tf$", {"regex string to identify TF files"}});
  options.push_back(ConfigParamSpec{"remote-regex", VariantType::String, "^(alien://|)/alice/data/.+", {"regex string to identify remote files"}}); // Use "^/eos/aliceo2/.+" for direct EOS access
  options.push_back(ConfigParamSpec{"max-cached-tf", VariantType::Int, 3, {"max TFs to cache in memory"}});
  options.push_back(ConfigParamSpec{"max-cached-files", VariantType::Int, 3, {"max TF files queued (copied for remote source)"}});
  options.push_back(ConfigParamSpec{"tf-reader-verbosity", VariantType::Int, 0, {"verbosity level (1 or 2: check RDH, print DH/DPH for 1st or all slices, >2 print RDH)"}});
  options.push_back(ConfigParamSpec{"raw-channel-config", VariantType::String, "", {"optional raw FMQ channel for non-DPL output"}});
  options.push_back(ConfigParamSpec{"send-diststf-0xccdb", VariantType::Bool, false, {"send explicit FLP/DISTSUBTIMEFRAME/0xccdb output"}});
  options.push_back(ConfigParamSpec{"disable-dummy-output", VariantType::Bool, false, {"Disable sending empty output if corresponding data is not found in the data"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});

  options.push_back(ConfigParamSpec{"timeframes-shm-limit", VariantType::String, "0", {"Minimum amount of SHM required in order to publish data"}});
  options.push_back(ConfigParamSpec{"metric-feedback-channel-format", VariantType::String, "name=metric-feedback,type=pull,method=connect,address=ipc://@metric-feedback-{},transport=shmem,rateLogging=0", {"format for the metric-feedback channel for TF rate limiting"}});

  // options for error-check suppression

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::rawdd::TFReaderInp rinp;
  rinp.inpdata = configcontext.options().get<std::string>("input-data");
  rinp.maxLoops = configcontext.options().get<int>("loop");
  int n = configcontext.options().get<int>("max-tf");
  rinp.maxTFs = n > 0 ? n : 0x7fffffff;
  rinp.detList = configcontext.options().get<std::string>("onlyDet");
  rinp.detListRawOnly = configcontext.options().get<std::string>("raw-only-det");
  rinp.detListNonRawOnly = configcontext.options().get<std::string>("non-raw-only-det");
  rinp.rawChannelConfig = configcontext.options().get<std::string>("raw-channel-config");
  rinp.delay_us = uint64_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds
  rinp.verbosity = configcontext.options().get<int>("tf-reader-verbosity");
  rinp.maxTFCache = std::max(1, configcontext.options().get<int>("max-cached-tf"));
  rinp.maxFileCache = std::max(1, configcontext.options().get<int>("max-cached-files"));
  rinp.copyCmd = configcontext.options().get<std::string>("copy-cmd");
  rinp.tffileRegex = configcontext.options().get<std::string>("tf-file-regex");
  rinp.remoteRegex = configcontext.options().get<std::string>("remote-regex");
  rinp.sendDummyForMissing = !configcontext.options().get<bool>("disable-dummy-output");
  rinp.sup0xccdb = !configcontext.options().get<bool>("send-diststf-0xccdb");
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  rinp.minSHM = std::stoul(configcontext.options().get<std::string>("timeframes-shm-limit"));
  int rateLimitingIPCID = std::stoi(configcontext.options().get<std::string>("timeframes-rate-limit-ipcid"));
  std::string chanFmt = configcontext.options().get<std::string>("metric-feedback-channel-format");
  if (rateLimitingIPCID > -1 && !chanFmt.empty()) {
    rinp.metricChannel = fmt::format(chanFmt, rateLimitingIPCID);
  }

  WorkflowSpec specs;
  specs.emplace_back(o2::rawdd::getTFReaderSpec(rinp));
  return specs;
}
