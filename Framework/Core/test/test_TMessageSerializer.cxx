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

  FairTMessage msg;
  TMessageSerializer::serialize(msg, &array);

  auto buf = as_span(msg);
  REQUIRE(buf.size() == msg.BufferSize());
  REQUIRE(static_cast<void*>(buf.data()) == static_cast<void*>(msg.Buffer()));
  // test deserialization with TObject as target class (default)
  auto out = TMessageSerializer::deserialize(buf);

  auto* outarr = dynamic_cast<TObjArray*>(out.get());
  REQUIRE(out.get() == outarr);
  auto* named = dynamic_cast<TNamed*>(outarr->At(0));
  REQUIRE(static_cast<void*>(named) == static_cast<void*>(outarr->At(0)));
  REQUIRE(named->GetName() == std::string(testname));
  REQUIRE(named->GetTitle() == std::string(testtitle));

  // test deserialization with a wrong target class and check the exception
  REQUIRE_THROWS_AS(TMessageSerializer::deserialize<TNamed>(buf), o2::framework::RuntimeErrorRef);

  REQUIRE_THROWS_MATCHES(TMessageSerializer::deserialize<TNamed>(buf), o2::framework::RuntimeErrorRef,
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

  FairTMessage msg;
  char* in = reinterpret_cast<char*>(&data);
  TMessageSerializer::serialize(msg, in, cl);

  auto out = TMessageSerializer::deserialize<std::vector<o2::test::Polymorphic>>(as_span(msg));
  REQUIRE(out);
  REQUIRE((*out.get()).size() == 2);
  REQUIRE((*out.get())[0] == o2::test::Polymorphic(0xaffe));
  REQUIRE((*out.get())[1] == o2::test::Polymorphic(0xd00f));

  // test deserialization with a wrong target class and check the exception
  REQUIRE_THROWS_AS(TMessageSerializer::deserialize(as_span(msg)), RuntimeErrorRef);
}

TEST_CASE("TestTMessageSerializer_InvalidBuffer")
{
  const char* buffer = "this is for sure not a serialized ROOT object";
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
  REQUIRE_THROWS_MATCHES(TMessageSerializer::deserialize<Dummy>((std::byte*)buffer, strlen(buffer)), o2::framework::RuntimeErrorRef, matcher);
}
