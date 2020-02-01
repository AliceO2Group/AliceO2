// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/ConfigParamSpec.h"
#include "Framework/DeviceSpec.h"
#include "Framework/AODReaderHelpers.h"

using namespace o2::framework;

void customize(std::vector<ConfigParamSpec>& options)
{
  //  options.push_back(ConfigParamSpec{"anInt", VariantType::Int, 1, {"an int option"}});
  //  options.push_back(ConfigParamSpec{"aFloat", VariantType::Float, 2.0f, {"a float option"}});
  //  options.push_back(ConfigParamSpec{"aDouble", VariantType::Double, 3., {"a double option"}});
  //  options.push_back(ConfigParamSpec{"aString", VariantType::String, "foo", {"a string option"}});
  //  options.push_back(ConfigParamSpec{"aBool", VariantType::Bool, true, {"a boolean option"}});
}

#include "Framework/runDataProcessing.h"

using namespace o2::framework;

// This is how you can define your processing in a declarative way
WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  std::vector<OutputSpec> outputs = {
    OutputSpec{"AOD", "TRACKPAR"},
    OutputSpec{"AOD", "TRACKPARCOV"},
    OutputSpec{"AOD", "TRACKEXTRA"},
    OutputSpec{"AOD", "CALO"},
    OutputSpec{"AOD", "MUON"},
    OutputSpec{"AOD", "VZERO"},
    OutputSpec{"AOD", "COLLISION"},
    OutputSpec{"AOD", "TIMEFRAME"}};
  int separateEnumerations = 0;
  DataProcessorSpec run2Converter{
    "esd-2-arrow-converter",
    {InputSpec{"enumeration",
               "DPL",
               "ENUM",
               static_cast<DataAllocator::SubSpecificationType>(separateEnumerations++), Lifetime::Enumeration}},
    outputs,
    readers::AODReaderHelpers::run2ESDConverterCallback(),
    {ConfigParamSpec{"esd-file", VariantType::String, "AliESDs.root", {"Input ESD file"}},
     ConfigParamSpec{"events", VariantType::Int, 0, {"Number of events to process (0 = all)"}},
     ConfigParamSpec{"start-value-enumeration", VariantType::Int64, 0ll, {"initial value for the enumeration"}},
     ConfigParamSpec{"end-value-enumeration", VariantType::Int64, -1ll, {"final value for the enumeration"}},
     ConfigParamSpec{"step-value-enumeration", VariantType::Int64, 1ll, {"step between one value and the other"}}}};

  return {run2Converter};
}
