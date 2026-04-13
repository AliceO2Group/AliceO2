// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef ALICEO2_ITSMFTALPIDEPARAM_INITIALIZER_H_
#define ALICEO2_ITSMFTALPIDEPARAM_INITIALIZER_H_
#include <vector>

namespace o2
{
namespace framework
{
class ConfigParamSpec;
class ConfigContext;
} // namespace framework
namespace itsmft
{

struct DPLAlpideParamInitializer {
  static constexpr char stagITSOpt[] = "enable-its-staggering";
  static constexpr char stagMFTOpt[] = "enable-mft-staggering";
  static constexpr bool stagDef = false;

  // DPL workflow options for staggering
  static void addConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts);
  static void addITSConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts);
  static bool isITSStaggeringEnabled(o2::framework::ConfigContext const& cfgc);
  static void addMFTConfigOption(std::vector<o2::framework::ConfigParamSpec>& opts);
  static bool isMFTStaggeringEnabled(o2::framework::ConfigContext const& cfgc);
};

} // namespace itsmft
} // namespace o2

#endif
