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

#include "Framework/TMessageSerializer.h"
#include "Framework/RuntimeError.h"
#include <fairmq/TransportFactory.h>
#include "TestClasses.h"
#include <catch_amalgamated.hpp>
#include <utility>

using namespace o2::framework;

class ExceptionMatcher : public Catch::Matchers::MatcherBase<RuntimeErrorRef>
{
  std::string m_expected;
  mutable std::string m_actual;

 public:
  ExceptionMatcher(std::string exp) : m_expected(std::move(exp)) {}
  bool match(RuntimeErrorRef const& ref) const override
  {
    auto& e = error_from_ref(ref);
    m_actual = std::string(e.what);
    return std::string(e.what) == m_expected;
  }
  std::string describe() const override
  {
    std::ostringstream ss;
    ss << " special exception has value of " << m_expected << " but got " << m_actual;
    return ss.str();
  }
};

TEST_CASE("TestTMessageSerializer")
{
  o2::framework::clean_all_runtime_errors();
  const char* testname = "testname";
  const char* testtitle = "testtitle";
  using namespace o2::framework;

  TObjArray array;
  array.SetOwner();
  array.Add(new TNamed(testname, testtitle));

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto msg = transport->CreateMessage(4096);
  FairOutputTBuffer buffer(*msg);
  TMessageSerializer::serialize(buffer, &array);

  FairInputTBuffer msg2((char*)msg->GetData(), msg->GetSize());
  // test deserialization with TObject as target class (default)
  auto out = TMessageSerializer::deserialize(msg2);

  auto* outarr = dynamic_cast<TObjArray*>(out.get());
  REQUIRE(out.get() == outarr);
  auto* named = dynamic_cast<TNamed*>(outarr->At(0));
  REQUIRE(static_cast<void*>(named) == static_cast<void*>(outarr->At(0)));
  REQUIRE(named->GetName() == std::string(testname));
  REQUIRE(named->GetTitle() == std::string(testtitle));

  // test deserialization with a wrong target class and check the exception
  REQUIRE_THROWS_AS(TMessageSerializer::deserialize<TNamed>(msg2), o2::framework::RuntimeErrorRef);

  REQUIRE_THROWS_MATCHES(TMessageSerializer::deserialize<TNamed>(msg2), o2::framework::RuntimeErrorRef,
                         ExceptionMatcher("can not convert serialized class TObjArray into target class TNamed"));
}

bool check_expected(RuntimeErrorRef const& ref)
{
  auto& e = error_from_ref(ref);
  std::string expected("can not convert serialized class vector<o2::test::Polymorphic> into target class TObject");
  return expected == e.what;
};

TEST_CASE("TestTMessageSerializer_NonTObject")
{
  using namespace o2::framework;
  std::vector<o2::test::Polymorphic> data{{0xaffe}, {0xd00f}};

  TClass* cl = TClass::GetClass("std::vector<o2::test::Polymorphic>");
  REQUIRE(cl != nullptr);

  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto msg = transport->CreateMessage(4096);
  FairOutputTBuffer buffer(*msg);
  char* in = reinterpret_cast<char*>(&data);
  TMessageSerializer::serialize(buffer, in, cl);
  FairInputTBuffer msg2((char*)msg->GetData(), msg->GetSize());

  auto out = TMessageSerializer::deserialize<std::vector<o2::test::Polymorphic>>(msg2);
  REQUIRE(out);
  REQUIRE((*out.get()).size() == 2);
  REQUIRE((*out.get())[0] == o2::test::Polymorphic(0xaffe));
  REQUIRE((*out.get())[1] == o2::test::Polymorphic(0xd00f));

  // test deserialization with a wrong target class and check the exception
  REQUIRE_THROWS_AS(TMessageSerializer::deserialize(msg2), RuntimeErrorRef);
}

TEST_CASE("TestTMessageSerializer_InvalidBuffer")
{
  const char* buffer = "this is for sure not a serialized ROOT object";
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto msg = transport->CreateMessage(strlen(buffer) + 8);
  memcpy((char*)msg->GetData() + 8, buffer, strlen(buffer));
  // test deserialization of invalid buffer and check the exception
  // FIXME: at the moment, TMessage fails directly with a segfault, which it shouldn't do
  /*
  try {
    auto out = TMessageSerializer::deserialize((std::byte*)buffer, strlen(buffer));
    BOOST_ERROR("here we should never get, the function call must fail with exception");
  } catch (std::exception& e) {
    std::string expected("");
    BOOST_CHECK_MESSAGE(expected == e.what(), e.what());
  }
  */
  // test deserialization of invalid target class and check the exception
  struct Dummy {
  };
  auto matcher = ExceptionMatcher("class is not ROOT-serializable: ZL22CATCH2_INTERNAL_TEST_4vE5Dummy");
  FairInputTBuffer msg2((char*)msg->GetData(), msg->GetSize());
  REQUIRE_THROWS_MATCHES(TMessageSerializer::deserialize<Dummy>(msg2), o2::framework::RuntimeErrorRef, matcher);
}

TEST_CASE("TestTMessageSerializer_CheckExpansion")
{
  const char* buffer = "this is for sure not a serialized ROOT object";
  auto transport = fair::mq::TransportFactory::CreateTransportFactory("zeromq");
  auto msg = transport->CreateMessage(strlen(buffer) + 8);
  FairOutputTBuffer msg2(*msg);
  // The buffer starts after 8 bytes.
  REQUIRE(msg2.Buffer() == (char*)msg->GetData() + 8);
  // The first 8 bytes of the buffer store the pointer to the message itself.
  REQUIRE(*(fair::mq::Message**)msg->GetData() == msg.get());
  // Notice that TBuffer does the same trick with the reallocation function,
  // so in the end the useful buffer size is the message size minus 16.
  REQUIRE(msg2.BufferSize() == (msg->GetSize() - 16));
  // This will not fit the original buffer size, so the buffer will be expanded.
  msg2.Expand(100);
}
