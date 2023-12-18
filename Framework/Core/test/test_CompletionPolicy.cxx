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

#include <catch_amalgamated.hpp>
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Framework/ServiceRegistry.h"
#include "Headers/DataHeader.h"
#include "Headers/NameHeader.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/InputSpan.h"
#include "Headers/Stack.h"

using namespace o2::framework;

TEST_CASE("TestCompletionPolicy")
{
  std::ostringstream oss;
  oss << CompletionPolicy::CompletionOp::Consume
      << CompletionPolicy::CompletionOp::Process
      << CompletionPolicy::CompletionOp::Wait
      << CompletionPolicy::CompletionOp::Discard;

  REQUIRE(oss.str() == "consumeprocesswaitdiscard");
}

TEST_CASE("TestCompletionPolicy_callback")
{
  o2::header::Stack stack{o2::header::DataHeader{"SOMEDATA", "TST", 0, 0}, o2::header::NameHeader<9>{"somename"}};

  auto matcher = [](auto const&) {
    return true;
  };

  ServiceRegistry services;

  auto callback = [&stack](InputSpan const& inputRefs, std::vector<InputSpec> const&, ServiceRegistryRef&) {
    for (auto const& ref : inputRefs) {
      auto const* header = CompletionPolicyHelpers::getHeader<o2::header::DataHeader>(ref);
      REQUIRE(header == reinterpret_cast<o2::header::DataHeader*>(stack.data()));
      REQUIRE(CompletionPolicyHelpers::getHeader<o2::header::NameHeader<9>>(ref) != nullptr);
      REQUIRE(CompletionPolicyHelpers::getHeader<o2::header::NameHeader<9>*>(ref) != nullptr);
    }
    return CompletionPolicy::CompletionOp::Consume;
  };

  std::vector<CompletionPolicy> policies{
    {"test", matcher, callback}};
  CompletionPolicy::InputSetElement ref{nullptr, reinterpret_cast<const char*>(stack.data()), nullptr};
  InputSpan const& inputs{[&ref](size_t) { return ref; }, 1};
  std::vector<InputSpec> specs;
  ServiceRegistryRef servicesRef{services};
  for (auto& policy : policies) {
    policy.callbackFull(inputs, specs, servicesRef);
  }
}
