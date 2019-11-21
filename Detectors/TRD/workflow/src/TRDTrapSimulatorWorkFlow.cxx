// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
#include "SimConfig/ConfigurableParam.h"

// for TRD
#include "TRDTrapSimulatorSpec.h"
#include "TRDTrackletWriterSpec.h"

// GRP   not sure if i need GRP, come back TODO
//#include "DataFormatsParameters/GRPObject.h"
//#include "GRPUpdaterSpec.h"

#include <cstdlib>
// this is somewhat assuming that a DPL workflow will run on one node
#include <thread> // to detect number of hardware threads
#include <string>
#include <sstream>
#include <cmath>
#include <unistd.h> // for getppid

using namespace o2::framework;

// ------------------------------------------------------------------

// customize the completion policy
/* void customize(std::vector<o2::framework::CompletionPolicy>& policies)
{
  using o2::framework::CompletionPolicy;
  // we customize the completion policy for the writer since it should stream immediately
  auto matcher = [](DeviceSpec const& device) {
    bool matched = device.name == "TPCDigitWriter";
    if (matched) {
      LOG(INFO) << "DPL completion policy for " << device.name << " customized";
    }
    return matched;
  };

  auto policy = [](gsl::span<o2::framework::PartRef const> const& inputs) {
    return CompletionPolicy::CompletionOp::Consume;
  };

  policies.push_back({CompletionPolicy{"process-any", matcher, policy}});
}*/

// ------------------------------------------------------------------

// we need to add workflow options before including Framework/runDataProcessing
// ------------------------------------------------------------------
 
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowoptions)
{
    //able to specify inputs
    //could be disk, upstream digitizer, run2 convert.
    //most of these are probably purely for debugging.
    //specify where the data is coming from i.e. ignore incoming message and use data as specified here, mostly for debugging as well.
    std::string filename;

    std::string trapsimindatahelp("Specify the location of incoming data for the simulator, full name of file");
    workflowoptions.push_back(ConfigParamSpec{"simdatasrc", VariantType::String, "none", {trapsimindatahelp}});

    //limit the trapsim to a specific roc or multiple rocs mostly for debugging.
    std::string trapsimrochelp("Specify the ROC to work on [0-540]");
    workflowoptions.push_back(ConfigParamSpec{"simROC", VariantType::Int,-1, {trapsimrochelp}});

    //limit to 1 supermodule.
    std::string trapsimsupermodulehelp("Specify the Supermodule to work on [0-18]");
    workflowoptions.push_back(ConfigParamSpec{"simSM", VariantType::Int, -1, {trapsimsupermodulehelp}});
    //limit to a stack in a supermodule
    std::string trapsimstackhelp("Specify the specific stack to work on [0-5] within the supermodule");
    workflowoptions.push_back(ConfigParamSpec{"simStack", VariantType::Int, -1, {trapsimstackhelp}});
     
    //probably more options to come.
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
  LOG(info) << "Input data file is : " << configcontext.options().get<std::string>("simdatasrc");
  LOG(info) << "simSm is : " << configcontext.options().get<int>("simSM");
  return WorkflowSpec{
    // connect the TRD digitization
    o2::trd::getTRDTrapSimulatorSpec(0),
    // connect the TRD digit writer
    o2::trd::getTRDTrackletWriterSpec()};

}

