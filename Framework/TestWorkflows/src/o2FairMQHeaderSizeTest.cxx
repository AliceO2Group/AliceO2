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

/// \file o2FairMQHeaderSizeTest.cxx
/// \brief Just a simple workflow to test how much messages can be stored internally,
///        when nothing is consumed. Used for tuning parameter shm-message-metadata-size.
///
/// \author Michal Tichak, michal.tichak@cern.ch

#include "Framework/ConfigParamSpec.h"
#include "Framework/ControlService.h"
#include "Framework/CallbackService.h"
#include "Framework/EndOfStreamContext.h"
#include "Framework/DeviceSpec.h"
#include "Framework/ControlService.h"
#include "Framework/runDataProcessing.h"

#include <chrono>
#include <thread>
#include <vector>
#include <random>

using namespace o2::framework;

static std::random_device rd;
static std::mt19937 gen(rd());

std::string random_string(size_t length)
{
  static const char alphanum[] =
    "0123456789"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz";

  std::uniform_int_distribution<> dis(0, sizeof(alphanum) - 2);

  std::string randomString;
  randomString.reserve(length);

  for (int i = 0; i < length; ++i) {
    randomString.push_back(alphanum[dis(gen)]);
  }

  return randomString;
}

std::string filename()
{
  std::stringstream ss;
  ss << "messages_count_" << random_string(10) << ".data";
  return std::move(ss).str();
}

WorkflowSpec defineDataProcessing(ConfigContext const& specs)
{
  return WorkflowSpec{
    {"A",
     Inputs{},
     {OutputSpec{{"a"}, "TST", "A"}},
     AlgorithmSpec{
       [numberOfMessages = 0, filename = filename()](ProcessingContext& ctx) mutable {
         using namespace std::chrono;
         ++numberOfMessages;
         // LOG(info) << "Generating message #" << ++numberOfMessages;

         {
           auto file = std::ofstream(filename, std::ios_base::out | std::ios_base::trunc);
           // file << duration_cast<milliseconds>(system_clock::now().time_since_epoch()).count() << "," << numberOfMessages << "\n";
           file << numberOfMessages;
         }

         auto& aData = ctx.outputs().make<int>(Output{"TST", "A", 0}, 1);
         aData[0] = 1;
       }}},
    {"B",
     {InputSpec{"x", "TST", "A"}},
     Outputs{},
     AlgorithmSpec{[](InitContext& ic) {
       return [](ProcessingContext& ctx) {
         while (true) {
           std::this_thread::sleep_for(std::chrono::milliseconds{100});
         }
         // auto& data = ctx.inputs().get<int>("x");
         // LOG(info) << "Reading message: " << data;
       };
     }}},
  };
}
