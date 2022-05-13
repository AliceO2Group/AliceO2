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

#include "DetectorsBase/Propagator.h"
#include "Framework/WorkflowSpec.h"
#include "Framework/ConfigParamSpec.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/DeviceSpec.h"
#include "Algorithm/RangeTokenizer.h"
#include "DetectorsCommonDataFormats/DetID.h"
#include "CommonUtils/ConfigurableParam.h"
#include "Framework/CompletionPolicyHelpers.h"

// for TRD
#include "TRDWorkflow/TRDTrapSimulatorSpec.h"
#include "TRDWorkflowIO/TRDTrackletWriterSpec.h"
#include "TRDWorkflowIO/TRDDigitReaderSpec.h"

#include "DataFormatsParameters/GRPObject.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h> // for getppid

using namespace o2::framework;

// ------------------------------------------------------------------
void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  // ordered policies for the writers
  policies.push_back(CompletionPolicyHelpers::consumeWhenAllOrdered(".*(?:TRD|trd).*[W,w]riter.*"));
}

void customize(std::vector<o2::framework::ConfigParamSpec>& workflowoptions)
{
  //able to specify inputs
  //could be disk, upstream digitizer, run2 convert.
  //most of these are probably purely for debugging.
  //specify where the data is coming from i.e. ignore incoming message and use data as specified here, mostly for debugging as well.
  std::string filename;

  //    std::string trapsimindatahelp("Specify the location of incoming data for the simulator, full name of file");
  //    workflowoptions.push_back(ConfigParamSpec{"simdatasrc", VariantType::String, "none", {trapsimindatahelp}});

  //limit the trapsim to a specific roc or multiple rocs mostly for debugging.
  std::string trapsimrochelp("Specify the ROC to work on [0-540]");
  workflowoptions.push_back(ConfigParamSpec{"simROC", VariantType::Int, -1, {trapsimrochelp}});

  //limit to 1 supermodule.
  std::string trapsimsupermodulehelp("Specify the Supermodule to work on [0-18]");
  workflowoptions.push_back(ConfigParamSpec{"simSM", VariantType::Int, -1, {trapsimsupermodulehelp}});
  //limit to a stack in a supermodule
  std::string trapsimstackhelp("Specify the specific stack to work on [0-5] within the supermodule");
  workflowoptions.push_back(ConfigParamSpec{"simStack", VariantType::Int, -1, {trapsimstackhelp}});

  workflowoptions.push_back(ConfigParamSpec{"disable-mc", o2::framework::VariantType::Bool, false, {"Disable MC labels"}});
  workflowoptions.push_back(ConfigParamSpec{"disable-root-input", o2::framework::VariantType::Bool, false, {"Disable root-files input readers"}});
  workflowoptions.push_back(ConfigParamSpec{"disable-root-output", o2::framework::VariantType::Bool, false, {"Disable root-files output writers"}});
  //limit to a stack in a supermodule
  // the next one is now done inside the trapsim spec.
  //  std::string trapsimconfighelp("Specify the Trap config to use from CCDB yes those long names like cf_pg-fpnp32_zs-s16-deh_tb24_trkl-b2p-fs1e24-ht200-qs0e24s24e23-pidlinear-pt100_ptrg.r5585");
  //  workflowoptions.push_back(ConfigParamSpec{"trapconfigname", VariantType::Int, -1, {trapsimconfighelp}});

  // option allowing to set parameters
  std::string keyvaluehelp("Semicolon separated key=value strings (e.g.: 'TRDSimParams.digithreads=4;...')");
  workflowoptions.push_back(
    ConfigParamSpec{"configKeyValues", VariantType::String, "", {keyvaluehelp}});
  workflowoptions.push_back(
    ConfigParamSpec{"configFile", VariantType::String, "", {"configuration file for configurable parameters"}});

  // json output
  // run2 input
  // trap configuration
  //
}

#include "Framework/runDataProcessing.h"

/// This function is required to be implemented to define the workflow
/// specifications
WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  // Reserve one entry which fill be filled with the SimReaderSpec
  // at the end. This places the processor at the beginning of the
  // workflow in the upper left corner of the GUI.
  //
  using namespace o2::conf;
  ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  WorkflowSpec specs;
  auto useMC = !configcontext.options().get<bool>("disable-mc");
  auto disableRootInput = configcontext.options().get<bool>("disable-root-input");
  auto disableRootOutput = configcontext.options().get<bool>("disable-root-output");
  if (!disableRootInput) {
    specs.emplace_back(o2::trd::getTRDDigitReaderSpec(useMC));
  }
  specs.emplace_back(o2::trd::getTRDTrapSimulatorSpec(useMC));
  if (!disableRootOutput) {
    specs.emplace_back(o2::trd::getTRDTrackletWriterSpec(useMC));
  }
  return specs;
}
