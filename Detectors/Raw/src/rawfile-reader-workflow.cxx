// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "RawFileReaderWorkflow.h"
#include "DetectorsRaw/RawFileReader.h"
#include "CommonUtils/ConfigurableParam.h"
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
  options.push_back(ConfigParamSpec{"loop", VariantType::Int, 1, {"loop N times (infinite for N<0)"}});
  options.push_back(ConfigParamSpec{"delay", VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"buffer-size", VariantType::Int64, 5 * 1024L, {"buffer size for files preprocessing"}});
  options.push_back(ConfigParamSpec{"super-page-size", VariantType::Int64, 1024L * 1024L, {"super-page size for FMQ parts definition"}});
  options.push_back(ConfigParamSpec{"part-per-hbf", VariantType::Bool, false, {"FMQ parts per superpage (default) of HBF"}});
  options.push_back(ConfigParamSpec{"raw-channel-config", VariantType::String, "", {"optional raw FMQ channel for non-DPL output"}});
  options.push_back(ConfigParamSpec{"cache-data", VariantType::Bool, false, {"cache data at 1st reading, may require excessive memory!!!"}});
  options.push_back(ConfigParamSpec{"detect-tf0", VariantType::Bool, false, {"autodetect HBFUtils start Orbit/BC from 1st TF seen"}});
  options.push_back(ConfigParamSpec{"calculate-tf-start", VariantType::Bool, false, {"calculate TF start instead of using TType"}});
  options.push_back(ConfigParamSpec{"drop-tf", VariantType::String, "none", {"Drop each TFid%(1)==(2) of detector, e.g. ITS,2,4;TPC,4[,0];..."}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});
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
  rinp.bufferSize = uint64_t(configcontext.options().get<int64_t>("buffer-size"));
  rinp.spSize = uint64_t(configcontext.options().get<int64_t>("super-page-size"));
  rinp.partPerSP = !configcontext.options().get<bool>("part-per-hbf");
  rinp.cache = configcontext.options().get<bool>("cache-data");
  rinp.autodetectTF0 = configcontext.options().get<bool>("detect-tf0");
  rinp.preferCalcTF = configcontext.options().get<bool>("calculate-tf-start");
  rinp.rawChannelConfig = configcontext.options().get<std::string>("raw-channel-config");
  rinp.delay_us = uint32_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds
  rinp.dropTF = configcontext.options().get<std::string>("drop-tf");
  rinp.errMap = 0;
  for (int i = RawFileReader::NErrorsDefined; i--;) {
    auto ei = RawFileReader::ErrTypes(i);
    bool defOpt = RawFileReader::ErrCheckDefaults[i];
    if (configcontext.options().get<bool>(RawFileReader::nochk_opt(ei).c_str()) ? !defOpt : defOpt) { // cmdl option inverts default!
      rinp.errMap |= 0x1 << i;
    }
  }
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));

  return std::move(o2::raw::getRawFileReaderWorkflow(rinp));
}
