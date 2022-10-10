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
#include "Framework/Plugins.h"
#include "Framework/AlgorithmSpec.h"
#include "CCDBHelpers.h"

struct CCDBFetcherPlugin : o2::framework::AlgorithmPlugin {
  o2::framework::AlgorithmSpec create() final
  {
    return o2::framework::CCDBHelpers::fetchFromCCDB();
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(CCDBFetcherPlugin, CustomAlgorithm);
DEFINE_DPL_PLUGINS_END
