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

#define BOOST_TEST_MODULE Test Framework AlgorithmSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/TMessageSerializer.h"
#include "TestClasses.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestTMessageSerializer)
{
  const char* testname = "testname";
  const char* testtitle = "testtitle";
  using namespace o2::framework;

  TObjArray array;
  array.SetOwner();
  array.Add(new TNamed(testname, testtitle));

  FairTMessage msg;
  TMessageSerializer::serialize(msg, &array);

  auto buf = as_span(msg);
  BOOST_CHECK_EQUAL(buf.size(), msg.BufferSize());
  BOOST_CHECK_EQUAL(static_cast<void*>(buf.data()), static_cast<void*>(msg.Buffer()));
  // test deserialization with TObject as target class (default)
  auto out = TMessageSerializer::deserialize(buf);

  TObjArray* outarr = dynamic_cast<TObjArray*>(out.get());
  BOOST_CHECK_EQUAL(out.get(), outarr);
  TNamed* named = dynamic_cast<TNamed*>(outarr->At(0));
  BOOST_CHECK_EQUAL(static_cast<void*>(named), static_cast<void*>(outarr->At(0)));
  BOOST_CHECK_EQUAL(named->GetName(), testname);
  BOOST_CHECK_EQUAL(named->GetTitle(), testtitle);

  // test deserialization with a wrong target class and check the exception
  BOOST_CHECK_EXCEPTION(TMessageSerializer::deserialize<TNamed>(buf),
                        RuntimeErrorRef,
                        [](RuntimeErrorRef const& ref) {
                          auto& e = error_from_ref(ref);
                          std::string expected("can not convert serialized class TObjArray into target class TNamed");
                          return expected == e.what;
                        });
}

bool check_expected(RuntimeErrorRef const& ref)
{
  auto& e = error_from_ref(ref);
  std::string expected("can not convert serialized class vector<o2::test::Polymorphic> into target class TObject");
  return expected == e.what;
};

BOOST_AUTO_TEST_CASE(TestTMessageSerializer_NonTObject)
{
  using namespace o2::framework;
  std::vector<o2::test::Polymorphic> data{{0xaffe}, {0xd00f}};

  TClass* cl = TClass::GetClass("std::vector<o2::test::Polymorphic>");
  BOOST_REQUIRE(cl != nullptr);

  FairTMessage msg;
  char* in = reinterpret_cast<char*>(&data);
  TMessageSerializer::serialize(msg, in, cl);

  auto out = TMessageSerializer::deserialize<std::vector<o2::test::Polymorphic>>(as_span(msg));
  BOOST_REQUIRE(out);
  BOOST_CHECK((*out.get()).size() == 2);
  BOOST_CHECK((*out.get())[0] == o2::test::Polymorphic(0xaffe));
  BOOST_CHECK((*out.get())[1] == o2::test::Polymorphic(0xd00f));

  // test deserialization with a wrong target class and check the exception
  BOOST_CHECK_EXCEPTION(TMessageSerializer::deserialize(as_span(msg)), RuntimeErrorRef, check_expected);
}

BOOST_AUTO_TEST_CASE(TestTMessageSerializer_InvalidBuffer)
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
  BOOST_CHECK_EXCEPTION(TMessageSerializer::deserialize<Dummy>((std::byte*)buffer, strlen(buffer)),
                        RuntimeErrorRef,
                        [](RuntimeErrorRef const& ref) {
                          auto& err = error_from_ref(ref);
                          return strcmp(err.what, "class is not ROOT-serializable") != 0;
                        });
}
