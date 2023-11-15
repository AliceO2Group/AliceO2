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

// @brief Aux.class initialize HBFUtils
// @author ruben.shahoyan@cern.ch

#ifndef _O2_HBFUTILS_INITIALIZER_
#define _O2_HBFUTILS_INITIALIZER_

#include <vector>
#include <string>
#include "CommonDataFormat/TFIDInfo.h"
#include "CommonUtils/NameConf.h"

namespace o2
{

namespace header
{
class DataHeader;
}
namespace framework
{
class ConfigContext;
class DataProcessorSpec;
class DataProcessingHeader;
class DeviceSpec;
class ConfigParamSpec;
class CallbacksPolicy;
using WorkflowSpec = std::vector<DataProcessorSpec>;
} // namespace framework

namespace raw
{

struct HBFUtilsInitializer {
  enum HBFOpt { NONE,
                INI,
                JSON,
                HBFUTILS,
                ROOT };
  static constexpr char DelayOpt[] = "reader-delay";
  static constexpr char HBFConfOpt[] = "hbfutils-config";
  static constexpr char HBFTFInfoOpt[] = "tf-info-source";
  static constexpr char HBFUSrc[] = "hbfutils";

  HBFUtilsInitializer(const o2::framework::ConfigContext& configcontext, o2::framework::WorkflowSpec& wf);
  static HBFOpt getOptType(const std::string& optString);
  static std::vector<o2::dataformats::TFIDInfo> readTFIDInfoVector(const std::string& fname);
  static void assignDataHeader(const std::vector<o2::dataformats::TFIDInfo>& tfinfoVec, o2::header::DataHeader& dh, o2::framework::DataProcessingHeader& dph);
  static void addNewTimeSliceCallback(std::vector<o2::framework::CallbacksPolicy>& policies);
  static void addConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts, const std::string& defOpt = std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE));
};

} // namespace raw
} // namespace o2

#endif
