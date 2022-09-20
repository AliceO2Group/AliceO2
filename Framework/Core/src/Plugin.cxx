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
//
#include "Framework/Plugins.h"
#include "Framework/ServiceSpec.h"
#include "Framework/CommonServices.h"

struct ThreadPoolPlugin : public o2::framework::ServicePlugin {
  o2::framework::ServiceSpec* create(void)
  {
    return o2::framework::CommonServices::threadPool(2);
  }
};

DEFINE_DPL_PLUGINS_BEGIN
DEFINE_DPL_PLUGIN_INSTANCE(ThreadPoolPlugin, CustomService);
DEFINE_DPL_PLUGINS_END
