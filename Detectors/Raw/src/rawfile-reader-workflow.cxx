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

using namespace o2::framework;
using namespace o2::raw;

// we need to add workflow options before including Framework/runDataProcessing
void customize(std::vector<o2::framework::ConfigParamSpec>& workflowOptions)
{
  // option allowing to set parameters
  std::vector<o2::framework::ConfigParamSpec> options;
  options.push_back(ConfigParamSpec{"conf", o2::framework::VariantType::String, "", {"configuration file to init from (obligatory)"}});
  options.push_back(ConfigParamSpec{"max-tf", o2::framework::VariantType::Int64, 0xffffffffL, {"max number of TF to process"}});
  options.push_back(ConfigParamSpec{"loop", o2::framework::VariantType::Int, 1, {"loop N times (infinite for N<0)"}});
  options.push_back(ConfigParamSpec{"message-per-tf", o2::framework::VariantType::Bool, false, {"send TF of each link as a single FMQ message rather than multipart with message per HB"}});
  options.push_back(ConfigParamSpec{"output-per-link", o2::framework::VariantType::Bool, false, {"send message per Link rather than per FMQ output route"}});
  options.push_back(ConfigParamSpec{"delay", o2::framework::VariantType::Float, 0.f, {"delay in seconds between consecutive TFs sending"}});
  options.push_back(ConfigParamSpec{"buffer-size", o2::framework::VariantType::Int64, 1024L * 1024L, {"buffer size for files preprocessing"}});
  options.push_back(ConfigParamSpec{"configKeyValues", VariantType::String, "", {"semicolon separated key=value strings"}});
  // options for error-check suppression
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrWrongPacketCounterIncrement), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrWrongPacketCounterIncrement)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrWrongPageCounterIncrement), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrWrongPageCounterIncrement)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrHBFStopOnFirstPage), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrHBFStopOnFirstPage)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrHBFNoStop), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrHBFNoStop)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrWrongFirstPage), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrWrongFirstPage)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrWrongHBFsPerTF), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrWrongHBFsPerTF)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrWrongNumberOfTF), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrWrongNumberOfTF)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrHBFJump), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrHBFJump)}});
  options.push_back(ConfigParamSpec{RawFileReader::nochk_opt(RawFileReader::ErrNoSuperPageForTF), VariantType::Bool, false, {RawFileReader::nochk_expl(RawFileReader::ErrNoSuperPageForTF)}});
  std::swap(workflowOptions, options);
}

// ------------------------------------------------------------------

#include "Framework/runDataProcessing.h"

WorkflowSpec defineDataProcessing(ConfigContext const& configcontext)
{
  auto inifile = configcontext.options().get<std::string>("conf");
  auto loop = configcontext.options().get<int>("loop");
  uint32_t maxTF = uint32_t(configcontext.options().get<int64_t>("max-tf"));
  uint64_t buffSize = uint64_t(configcontext.options().get<int64_t>("buffer-size"));
  auto tfAsMessage = configcontext.options().get<bool>("message-per-tf");
  auto outPerRoute = !configcontext.options().get<bool>("output-per-link");
  o2::conf::ConfigurableParam::updateFromString(configcontext.options().get<std::string>("configKeyValues"));
  uint32_t delay_us = uint32_t(1e6 * configcontext.options().get<float>("delay")); // delay in microseconds

  uint32_t errmap = 0xffffffff;
  for (int i = RawFileReader::NErrorsDefined; i--;) {
    if (configcontext.options().get<bool>(RawFileReader::nochk_opt(RawFileReader::ErrTypes(i)).c_str())) {
      errmap ^= 0x1 << i;
      LOGF(INFO, "ignore  check for /%s/", RawFileReader::ErrNames[i].data());
    }
  }

  return std::move(o2::raw::getRawFileReaderWorkflow(inifile, tfAsMessage, outPerRoute, loop, delay_us, errmap, maxTF, buffSize));
}
