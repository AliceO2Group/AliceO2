// Copyright 2019-2025 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#include "Framework/Plugins.h"
#include "Framework/AlgorithmSpec.h"
#include "AODReaderHelpers.h"

struct ExtendedTableSpawner : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create(o2::framework::ConfigContext const& config) override
  {
    return o2::framework::readers::AODReaderHelpers::aodSpawnerCallback(config);
  }
};

struct IndexTableBuilder : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create(o2::framework::ConfigContext const& config) override
  {
    return o2::framework::readers::AODReaderHelpers::indexBuilderCallback(config);
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(ExtendedTableSpawner, CustomAlgorithm);
DEFINE_DPL_PLUGIN_INSTANCE(IndexTableBuilder, CustomAlgorithm);
DEFINE_DPL_PLUGINS_END
