// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <string>
#include <vector>
#include "Framework/Logger.h"
#include "Framework/ControlService.h"
#include "Framework/ConfigParamRegistry.h"
#include "Framework/InputSpec.h"
#include "DetectorsCommonDataFormats/NameConf.h"
#include "CTFWorkflow/CTFReaderSpec.h"
#include "DataFormatsParameters/GRPObject.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"

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
#include "MCHWorkflow/EntropyDecoderSpec.h"
#include "EMCALWorkflow/EntropyDecoderSpec.h"
#include "PHOSWorkflow/EntropyDecoderSpec.h"
#include "CPVWorkflow/EntropyDecoderSpec.h"
#include "ZDCWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, std::string{DetID::NONE}, {"comma-separated list of detectors to accept. Overrides skipDet"}});
  options.push_back(ConfigParamSpec{"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma-separate list of detectors to skip"}});
  options.push_back(ConfigParamSpec{"ctf-input", VariantType::String, "none", {"comma-separated list CTF input files"}});
  options.push_back(ConfigParamSpec{"loop", VariantType::Int, 1, {"loop N times (infinite for N<=0)"}});
  options.push_back(ConfigParamSpec{"delay", VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"Semicolon separated key=value strings"}});

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  DetID::mask_t dets;
  dets.set(); // by default read all
  WorkflowSpec specs;

  auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
  auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
  if (mskOnly.any()) {
    dets &= mskOnly;
  } else {
    dets ^= mskSkip;
  }

  std::string inpNames = configcontext.options().get<std::string>("ctf-input");
  if (inpNames.empty() || inpNames == "none") {
    if (!configcontext.helpOnCommandLine()) {
      throw std::runtime_error("--ctf-input <file,...> is not provided");
    }
    inpNames = "";
  }

  int loop = configcontext.options().get<int>("loop");
  if (loop < 1) {
    loop = 0x7fffffff;
  }
  int delayMUS = int32_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds
  if (delayMUS < 0) {
    delayMUS = 0;
  }

  specs.push_back(o2::ctf::getCTFReaderSpec(dets, inpNames, loop, delayMUS));

  // add decodors for all allowed detectors.
  if (dets[DetID::ITS]) {
    specs.push_back(o2::itsmft::getEntropyDecoderSpec(DetID::getDataOrigin(DetID::ITS)));
  }
  if (dets[DetID::MFT]) {
    specs.push_back(o2::itsmft::getEntropyDecoderSpec(DetID::getDataOrigin(DetID::MFT)));
  }
  if (dets[DetID::TPC]) {
    specs.push_back(o2::tpc::getEntropyDecoderSpec());
  }
  if (dets[DetID::TRD]) {
    specs.push_back(o2::trd::getEntropyDecoderSpec());
  }
  if (dets[DetID::TOF]) {
    specs.push_back(o2::tof::getEntropyDecoderSpec());
  }
  if (dets[DetID::FT0]) {
    specs.push_back(o2::ft0::getEntropyDecoderSpec());
  }
  if (dets[DetID::FV0]) {
    specs.push_back(o2::fv0::getEntropyDecoderSpec());
  }
  if (dets[DetID::FDD]) {
    specs.push_back(o2::fdd::getEntropyDecoderSpec());
  }
  if (dets[DetID::MID]) {
    specs.push_back(o2::mid::getEntropyDecoderSpec());
  }
  if (dets[DetID::MCH]) {
    specs.push_back(o2::mch::getEntropyDecoderSpec());
  }
  if (dets[DetID::EMC]) {
    specs.push_back(o2::emcal::getEntropyDecoderSpec());
  }
  if (dets[DetID::PHS]) {
    specs.push_back(o2::phos::getEntropyDecoderSpec());
  }
  if (dets[DetID::CPV]) {
    specs.push_back(o2::cpv::getEntropyDecoderSpec());
  }
  if (dets[DetID::ZDC]) {
    specs.push_back(o2::zdc::getEntropyDecoderSpec());
  }
  if (dets[DetID::HMP]) {
    specs.push_back(o2::hmpid::getEntropyDecoderSpec());
  }

  return std::move(specs);
}
