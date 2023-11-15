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

#include "ITStracking/TrackingConfigParam.h"
#include "ITStracking/Configuration.h"

namespace o2
{
namespace its
{
static auto& sVertexerParamITS = o2::its::VertexerParamConfig::Instance();
static auto& sCATrackerParamITS = o2::its::TrackerParamConfig::Instance();
static auto& sGpuRecoParamITS = o2::its::GpuRecoParamConfig::Instance();

O2ParamImpl(o2::its::VertexerParamConfig);
O2ParamImpl(o2::its::TrackerParamConfig);
O2ParamImpl(o2::its::GpuRecoParamConfig);
} // namespace its
} // namespace o2
