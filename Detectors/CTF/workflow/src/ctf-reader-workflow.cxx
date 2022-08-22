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

#include <string>
#include <vector>
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/ChannelSpecHelpers.h"
#include "Framework/InputSpec.h"
#include "CommonUtils/NameConf.h"
#include "CTFWorkflow/CTFReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Algorithm/RangeTokenizer.h"

// Specific detectors specs
#include "ITSMFTWorkflow/EntropyDecoderSpec.h"
#include "TPCWorkflow/EntropyDecoderSpec.h"
#include "TRDWorkflow/EntropyDecoderSpec.h"
#include "HMPIDWorkflow/EntropyDecoderSpec.h"
#include "FT0Workflow/EntropyDecoderSpec.h"
#include "FV0Workflow/EntropyDecoderSpec.h"
#include "FDDWorkflow/EntropyDecoderSpec.h"
#include "TOFWorkflowUtils/EntropyDecoderSpec.h"
#include "MIDWorkflow/EntropyDecoderSpec.h"
#include "MCHCTF/EntropyDecoderSpec.h"
#include "EMCALWorkflow/EntropyDecoderSpec.h"
#include "PHOSWorkflow/EntropyDecoderSpec.h"
#include "CPVWorkflow/EntropyDecoderSpec.h"
#include "ZDCWorkflow/EntropyDecoderSpec.h"
#include "CTPWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"ctf-input", VariantType::String, "none", {"comma-separated list CTF input files"}});
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, std::string{DetID::ALL}, {"comma-separated list of detectors to accept. Overrides skipDet"}});
  options.push_back(ConfigParamSpec{"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma-separate list of detectors to skip"}});
  options.push_back(ConfigParamSpec{"max-tf", VariantType::Int, -1, {"max CTFs to process (<= 0 : infinite)"}});
  options.push_back(ConfigParamSpec{"loop", VariantType::Int, 0, {"loop N times (infinite for N<0)"}});
  options.push_back(ConfigParamSpec{"delay", VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"copy-cmd", VariantType::String, "alien_cp ?src file://?dst", {"copy command for remote files or no-copy to avoid copying"}}); // Use "XrdSecPROTOCOL=sss,unix xrdcp -N root://eosaliceo2.cern.ch/?src ?dst" for direct EOS access
  options.push_back(ConfigParamSpec{"ctf-file-regex", VariantType::String, ".*o2_ctf_run.+\\.root$", {"regex string to identify CTF files"}});
  options.push_back(ConfigParamSpec{"remote-regex", VariantType::String, "^(alien://|)/alice/data/.+", {"regex string to identify remote files"}}); // Use "^/eos/aliceo2/.+" for direct EOS access
  options.push_back(ConfigParamSpec{"max-cached-files", VariantType::Int, 3, {"max CTF files queued (copied for remote source)"}});
  options.push_back(ConfigParamSpec{"allow-missing-detectors", VariantType::Bool, false, {"send empty message if detector is missing in the CTF (otherwise throw)"}});
  options.push_back(ConfigParamSpec{"send-diststf-0xccdb", VariantType::Bool, false, {"send explicit FLP/DISTSUBTIMEFRAME/0xccdb output"}});
  options.push_back(ConfigParamSpec{"ctf-reader-verbosity", VariantType::Int, 0, {"verbosity level (0: summary per detector, 1: summary per block"}});
  options.push_back(ConfigParamSpec{"ctf-data-subspec", VariantType::Int, 0, {"subspec to use for decoded CTF messages (use non-0 if CTF writer will be attached downstream)"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});
  //
  options.push_back(ConfigParamSpec{"its-digits", VariantType::Bool, false, {"convert ITS clusters to digits"}});
  options.push_back(ConfigParamSpec{"mft-digits", VariantType::Bool, false, {"convert MFT clusters to digits"}});

  options.push_back(ConfigParamSpec{"timeframes-shm-limit", VariantType::String, "0", {"Minimum amount of SHM required in order to publish data"}});
  options.push_back(ConfigParamSpec{"metric-feedback-channel-format", VariantType::String, "name=metric-feedback,type=pull,method=connect,address=ipc://{}metric-feedback-{},transport=shmem,rateLogging=0", {"format for the metric-feedback channel for TF rate limiting"}});
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  o2::ctf::CTFReaderInp ctfInput;

  WorkflowSpec specs;
  std::string allowedDetectors = "ITS,TPC,TRD,TOF,PHS,CPV,EMC,HMP,MFT,MCH,MID,ZDC,FT0,FV0,FDD,CTP"; // FIXME: explicit list to avoid problem with upgrade detectors
  auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
  auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
  if (mskOnly.any()) {
    ctfInput.detMask &= mskOnly;
  } else {
    ctfInput.detMask ^= mskSkip;
  }
  ctfInput.detMask &= DetID::getMask(allowedDetectors);
  ctfInput.inpdata = configcontext.options().get<std::string>("ctf-input");
  ctfInput.subspec = (unsigned int)configcontext.options().get<int>("ctf-data-subspec");
  if (ctfInput.inpdata.empty() || ctfInput.inpdata == "none") {
    if (!configcontext.helpOnCommandLine()) {
      throw std::runtime_error("--ctf-input <file,...> is not provided");
    }
    ctfInput.inpdata = "";
  }

  ctfInput.maxLoops = configcontext.options().get<int>("loop");
  if (ctfInput.maxLoops < 0) {
    ctfInput.maxLoops = 0x7fffffff;
  }
  ctfInput.delay_us = int32_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds
  if (ctfInput.delay_us < 0) {
    ctfInput.delay_us = 0;
  }
  int n = configcontext.options().get<int>("max-tf");
  ctfInput.maxTFs = n > 0 ? n : 0x7fffffff;

  ctfInput.maxFileCache = std::max(1, configcontext.options().get<int>("max-cached-files"));

  ctfInput.copyCmd = configcontext.options().get<std::string>("copy-cmd");
  ctfInput.tffileRegex = configcontext.options().get<std::string>("ctf-file-regex");
  ctfInput.remoteRegex = configcontext.options().get<std::string>("remote-regex");
  ctfInput.allowMissingDetectors = configcontext.options().get<bool>("allow-missing-detectors");
  ctfInput.sup0xccdb = !configcontext.options().get<bool>("send-diststf-0xccdb");
  ctfInput.minSHM = std::stoul(configcontext.options().get<std::string>("timeframes-shm-limit"));
  int verbosity = configcontext.options().get<int>("ctf-reader-verbosity");

  int rateLimitingIPCID = std::stoi(configcontext.options().get<std::string>("timeframes-rate-limit-ipcid"));
  std::string chanFmt = configcontext.options().get<std::string>("metric-feedback-channel-format");
  if (rateLimitingIPCID > -1 && !chanFmt.empty()) {
    ctfInput.metricChannel = fmt::format(chanFmt, o2::framework::ChannelSpecHelpers::defaultIPCFolder(), rateLimitingIPCID);
  }

  specs.push_back(o2::ctf::getCTFReaderSpec(ctfInput));

  // add decoders for all allowed detectors.
  if (ctfInput.detMask[DetID::ITS]) {
    specs.push_back(o2::itsmft::getEntropyDecoderSpec(DetID::getDataOrigin(DetID::ITS), verbosity, configcontext.options().get<bool>("its-digits"), ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::MFT]) {
    specs.push_back(o2::itsmft::getEntropyDecoderSpec(DetID::getDataOrigin(DetID::MFT), verbosity, configcontext.options().get<bool>("mft-digits"), ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::TPC]) {
    specs.push_back(o2::tpc::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::TRD]) {
    specs.push_back(o2::trd::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::TOF]) {
    specs.push_back(o2::tof::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::FT0]) {
    specs.push_back(o2::ft0::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::FV0]) {
    specs.push_back(o2::fv0::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::FDD]) {
    specs.push_back(o2::fdd::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::MID]) {
    specs.push_back(o2::mid::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::MCH]) {
    specs.push_back(o2::mch::getEntropyDecoderSpec(verbosity, "mch-entropy-decoder", ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::EMC]) {
    specs.push_back(o2::emcal::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::PHS]) {
    specs.push_back(o2::phos::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::CPV]) {
    specs.push_back(o2::cpv::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::ZDC]) {
    specs.push_back(o2::zdc::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::HMP]) {
    specs.push_back(o2::hmpid::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }
  if (ctfInput.detMask[DetID::CTP]) {
    specs.push_back(o2::ctp::getEntropyDecoderSpec(verbosity, ctfInput.subspec));
  }

  return std::move(specs);
}
