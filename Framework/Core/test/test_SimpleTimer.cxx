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
#include "Framework/AlgorithmSpec.h"
#include "Framework/runDataProcessing.h"
#include "Framework/ControlService.h"

#include <chrono>
#include <thread>

using namespace o2::framework;

// This is how you can define your processing in a declarative way
std::vector<DataProcessorSpec> defineDataProcessing(ConfigContext const&)
{
  return {
    DataProcessorSpec{
      "enumeration",
      Inputs{},
      {},
      AlgorithmSpec{
        adaptStateless([](ControlService& control) {
          // This is invoked autonomously by the timer.
          std::this_thread::sleep_for(std::chrono::seconds(1));
          control.readyToQuit(QuitRequest::All);
        })}},
    DataProcessorSpec{
      "atimer",
      Inputs{
        InputSpec{"atimer", "TST", "TIMER", 0, Lifetime::Timer}},
      {},
      AlgorithmSpec{
        adaptStateless([](ControlService& control) {
          // This is invoked autonomously by the timer.
          control.readyToQuit(QuitRequest::Me);
        })}},
    DataProcessorSpec{
      "btimer",
      Inputs{
        InputSpec{"btimer", "TST", "TIMER2", 0, Lifetime::Timer}},
      {},
      AlgorithmSpec{
        adaptStateless([](ControlService& control) {
          // This is invoked autonomously by the timer.
          control.readyToQuit(QuitRequest::Me);
        })},
      {ConfigParamSpec{"period-btimer", VariantType::Int, 2000, {"period of timer"}}}}};
}
