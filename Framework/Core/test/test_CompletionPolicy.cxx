// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework CompletionPolicy
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/CompletionPolicy.h"
#include "Framework/CompletionPolicyHelpers.h"
#include "Headers/DataHeader.h"
#include "Headers/NameHeader.h"
#include "Framework/CompletionPolicy.h"
#include "Framework/InputSpan.h"
#include "Headers/Stack.h"

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestCompletionPolicy)
{
  std::ostringstream oss;
  oss << CompletionPolicy::CompletionOp::Consume
      << CompletionPolicy::CompletionOp::Process
      << CompletionPolicy::CompletionOp::Wait
      << CompletionPolicy::CompletionOp::Discard;

  BOOST_REQUIRE_EQUAL(oss.str(), "consumeprocesswaitdiscard");
}

BOOST_AUTO_TEST_CASE(TestCompletionPolicy_callback)
{
  o2::header::Stack stack{o2::header::DataHeader{"SOMEDATA", "TST", 0, 0}, o2::header::NameHeader<9>{"somename"}};

  auto matcher = [](auto const&) {
    return true;
  };

  auto callback = [&stack](InputSpan const& inputRefs) {
    for (auto const& ref : inputRefs) {
      auto const* header = CompletionPolicyHelpers::getHeader<o2::header::DataHeader>(ref);
      BOOST_CHECK_EQUAL(header, reinterpret_cast<o2::header::DataHeader*>(stack.data()));
      BOOST_CHECK(CompletionPolicyHelpers::getHeader<o2::header::NameHeader<9>>(ref) != nullptr);
      BOOST_CHECK(CompletionPolicyHelpers::getHeader<o2::header::NameHeader<9>*>(ref) != nullptr);
    }
    return CompletionPolicy::CompletionOp::Consume;
  };

  std::vector<CompletionPolicy> policies;
  policies.emplace_back("test", matcher, callback);

  CompletionPolicy::InputSetElement ref{nullptr, reinterpret_cast<const char*>(stack.data()), nullptr};
  InputSpan const& inputs{[&ref](size_t) { return ref; }, 1};
  for (auto& policy : policies) {
    policy.callback(inputs);
  }
}
