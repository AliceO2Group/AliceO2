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
#include "Framework/DataRefUtils.h"
#include "Framework/runDataProcessing.h"
#include "Framework/Monitoring.h"
#include "Framework/Logger.h"
#include "Framework/TableBuilder.h"
#include <TH1F.h>

#include <chrono>
#include <vector>

using Monitoring = o2::monitoring::Monitoring;
using namespace o2::framework;

using DataHeader = o2::header::DataHeader;
using DataOrigin = o2::header::DataOrigin;

std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  DataProcessorSpec timeframeReader{
    "reader",
    Inputs{},
    {OutputSpec{{"tpc"}, "TPC", "CLUSTERS"}},
    AlgorithmSpec{
      [](ProcessingContext& ctx) {
        std::this_thread::sleep_for(std::chrono::seconds(1));

        auto builder = ctx.outputs().make<TableBuilder>(Output{"TPC", "CLUSTERS", 0});
        auto rowWriter = builder->persist<float, float, float>({"x", "y", "z"});
        for (size_t i = 0; i < 3; ++i) {
          rowWriter(0, 0.f, 0.f, 0.f);
        }
        builder->finalize();
      }}};

  return {timeframeReader};
}
