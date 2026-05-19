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

/// \file CorrectionMapsOptions.h
/// \brief Helper class to parse options for correction maps
/// \author matthias.kleiner@cern.ch

#ifndef TPC_CORRECTION_MAPS_OPTIONS_H_
#define TPC_CORRECTION_MAPS_OPTIONS_H_

#ifndef GPUCA_GPUCODE_DEVICE
#include <memory>
#include <vector>
#endif
#include "CorrectionMapsTypes.h"

namespace o2
{
namespace framework
{
class ConfigParamRegistry;
class ConfigParamSpec;
} // namespace framework

namespace tpc
{

class CorrectionMapsOptions
{
 public:
  CorrectionMapsOptions() = default;
  ~CorrectionMapsOptions() = default;
  CorrectionMapsOptions(const CorrectionMapsOptions&) = delete;

#ifndef GPUCA_GPUCODE_DEVICE
  static CorrectionMapsGloOpts parseGlobalOptions(const o2::framework::ConfigParamRegistry& opts);
  static void addGlobalOptions(std::vector<o2::framework::ConfigParamSpec>& options);

 protected:
  static void addOption(std::vector<o2::framework::ConfigParamSpec>& options, o2::framework::ConfigParamSpec&& osp);
#endif
};

} // namespace tpc

} // namespace o2

#endif
