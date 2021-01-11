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

// Specific detectors specs
#include "ITSMFTWorkflow/EntropyDecoderSpec.h"
#include "TPCWorkflow/EntropyDecoderSpec.h"
#include "FT0Workflow/EntropyDecoderSpec.h"
#include "FV0Workflow/EntropyDecoderSpec.h"
#include "FDDWorkflow/EntropyDecoderSpec.h"
#include "TOFWorkflowUtils/EntropyDecoderSpec.h"
#include "MIDWorkflow/EntropyDecoderSpec.h"
#include "EMCALWorkflow/EntropyDecoderSpec.h"
#include "PHOSWorkflow/EntropyDecoderSpec.h"
#include "CPVWorkflow/EntropyDecoderSpec.h"

using namespace o2::framework;
using DetID = o2::detectors::DetID;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"onlyDet", VariantType::String, std::string{DetID::NONE}, {"comma-separated list of detectors to accept. Overrides skipDet"}});
  options.push_back(ConfigParamSpec{"skipDet", VariantType::String, std::string{DetID::NONE}, {"comma-separate list of detectors to skip"}});
  options.push_back(ConfigParamSpec{"ctf-input", VariantType::String, "", {"comma-separated list CTF input files"}});

  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------
#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  DetID::mask_t dets;
  WorkflowSpec specs;

  std::string inpNames;
  if (!configcontext.helpOnCommandLine()) {
    dets.set(); // by default read all
    auto mskOnly = DetID::getMask(configcontext.options().get<std::string>("onlyDet"));
    auto mskSkip = DetID::getMask(configcontext.options().get<std::string>("skipDet"));
    if (mskOnly.any()) {
      dets &= mskOnly;
    } else {
      dets ^= mskSkip;
    }
    if ((inpNames = configcontext.options().get<std::string>("ctf-input")).empty()) {
      throw std::runtime_error("--ctf-input <file,...> is not provided");
    }
  }

  specs.push_back(o2::ctf::getCTFReaderSpec(dets, inpNames));
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
  if (dets[DetID::EMC]) {
    specs.push_back(o2::emcal::getEntropyDecoderSpec());
  }
  if (dets[DetID::PHS]) {
    specs.push_back(o2::phos::getEntropyDecoderSpec());
  }
  if (dets[DetID::CPV]) {
    specs.push_back(o2::cpv::getEntropyDecoderSpec());
  }

  return std::move(specs);
}
