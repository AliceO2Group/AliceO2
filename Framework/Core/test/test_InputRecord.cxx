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
#include "Framework/InputRecord.h"
#include "Framework/InputSpan.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

bool any_exception(RuntimeErrorRef const&) { return true; }

TEST_CASE("TestInputRecord")
{
  // Create the routes we want for the InputRecord
  InputSpec spec1{"x", "TPC", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec2{"y", "ITS", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec3{"z", "TST", "EMPTY", 0, Lifetime::Timeframe};

  size_t i = 0;
  auto createRoute = [&i](const char* source, InputSpec& spec) {
    return InputRoute{
      spec,
      i++,
      source,
      0,
      std::nullopt};
  };

  /// FIXME: keep it simple and simply use the constructor...
  std::vector<InputRoute> schema = {
    createRoute("x_source", spec1),
    createRoute("y_source", spec2),
    createRoute("z_source", spec3)};
  // First of all we test if an empty registry behaves as expected, raising a
  // bunch of exceptions.
  InputSpan span{
    [](size_t) { return DataRef{nullptr, nullptr, nullptr}; },
    0};
  ServiceRegistry registry;
  InputRecord emptyRecord(schema, span, registry);

  REQUIRE_THROWS_AS(emptyRecord.get("x"), RuntimeErrorRef);
  REQUIRE_THROWS_AS(emptyRecord.get("y"), RuntimeErrorRef);
  REQUIRE_THROWS_AS(emptyRecord.get("z"), RuntimeErrorRef);
  REQUIRE_THROWS_AS((void)emptyRecord.getByPos(0), RuntimeErrorRef);
  REQUIRE_THROWS_AS((void)emptyRecord.getByPos(1), RuntimeErrorRef);
  REQUIRE_THROWS_AS((void)emptyRecord.getByPos(2), RuntimeErrorRef);
  // Then we actually check with a real set of inputs.

  std::vector<void*> inputs;

  auto createMessage = [&inputs](DataHeader& dh, int value) {
    DataProcessingHeader dph{0, 1};
    Stack stack{dh, dph};
    void* header = malloc(stack.size());
    void* payload = malloc(sizeof(int));
    memcpy(header, stack.data(), stack.size());
    memcpy(payload, &value, sizeof(int));
    inputs.emplace_back(header);
    inputs.emplace_back(payload);
  };

  auto createEmpty = [&inputs]() {
    inputs.emplace_back(nullptr);
    inputs.emplace_back(nullptr);
  };

  DataHeader dh1;
  dh1.dataDescription = "CLUSTERS";
  dh1.dataOrigin = "TPC";
  dh1.subSpecification = 0;
  dh1.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  DataHeader dh2;
  dh2.dataDescription = "CLUSTERS";
  dh2.dataOrigin = "ITS";
  dh2.subSpecification = 0;
  dh2.payloadSerializationMethod = o2::header::gSerializationMethodNone;
  createMessage(dh1, 1);
  createMessage(dh2, 2);
  createEmpty();
  InputSpan span2{[&inputs](size_t i) { return DataRef{nullptr, static_cast<char const*>(inputs[2 * i]), static_cast<char const*>(inputs[2 * i + 1])}; }, inputs.size() / 2};
  InputRecord record{schema, span2, registry};

  // Checking we can get the whole ref by name
  REQUIRE_NOTHROW(record.get("x"));
  REQUIRE_NOTHROW(record.get("y"));
  REQUIRE_NOTHROW(record.get("z"));
  auto ref00 = record.get("x");
  auto ref10 = record.get("y");
  REQUIRE_THROWS_AS(record.get("err"), RuntimeErrorRef);

  // Or we can get it positionally
  REQUIRE_NOTHROW(record.get("x"));
  auto ref01 = record.getByPos(0);
  auto ref11 = record.getByPos(1);
  REQUIRE_THROWS_AS((void)record.getByPos(10), RuntimeErrorRef);

  // This should be exactly the same pointers
  REQUIRE(ref00.header == ref01.header);
  REQUIRE(ref00.payload == ref01.payload);
  REQUIRE(ref10.header == ref11.header);
  REQUIRE(ref10.payload == ref11.payload);

  REQUIRE(record.isValid("x") == true);
  REQUIRE(record.isValid("y") == true);
  REQUIRE(record.isValid("z") == false);
  REQUIRE(record.size() == 3);
  REQUIRE(record.countValidInputs() == 2);

  REQUIRE(record.isValid(0) == true);
  REQUIRE(record.isValid(1) == true);
  REQUIRE(record.isValid(2) == false);
  // This by default is a shortcut for
  //
  // *static_cast<int const *>(record.get("x").payload);
  //
  REQUIRE(record.get<int>("x") == 1);
  REQUIRE(record.get<int>("y") == 2);
  // A few more time just to make sure we are not stateful..
  REQUIRE(record.get<int>("x") == 1);
  REQUIRE(record.get<int>("x") == 1);

  // test the iterator
  int position = 0;
  for (auto input = record.begin(), end = record.end(); input != end; input++, position++) {
    if (position == 0) {
      REQUIRE(input.matches("TPC") == true);
      REQUIRE(input.matches("TPC", "CLUSTERS") == true);
      REQUIRE(input.matches("ITS", "CLUSTERS") == false);
    }
    if (position == 1) {
      REQUIRE(input.matches("ITS") == true);
      REQUIRE(input.matches("ITS", "CLUSTERS") == true);
      REQUIRE(input.matches("TPC", "CLUSTERS") == false);
    }
    // check if invalid slots are filtered out by the iterator
    REQUIRE(position != 2);
  }

  // the 2-level iterator to access inputs and their parts
  // all inputs have 1 part, we check the first input
  REQUIRE(record.begin().size() == 1);
  // the end-instance of the inputs has no parts
  REQUIRE(record.end().size() == 0);
  // thus there is no element and begin == end
  REQUIRE(record.end().begin() == record.end().end());
}

// TODO:
// - test all `get` implementations
// - create a list of supported types and check that the API compiles
// - test return value optimization for vectors, unique_ptr
// - check objects which work directly on the payload for zero-copy
