// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework InputRecord
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/InputRecord.h"
#include "Framework/DataProcessingHeader.h"
#include "Headers/DataHeader.h"
#include "Headers/Stack.h"

using namespace o2::framework;
using DataHeader = o2::header::DataHeader;
using Stack = o2::header::Stack;

bool any_exception(std::exception const& ex) { return true; }

BOOST_AUTO_TEST_CASE(TestInputRecord)
{
  // Create the routes we want for the InputRecord
  InputSpec spec1{"x", "TPC", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec2{"y", "ITS", "CLUSTERS", 0, Lifetime::Timeframe};
  InputSpec spec3{"z", "TST", "EMPTY", 0, Lifetime::Timeframe};

  auto createRoute = [](const char* source, InputSpec& spec) {
    return InputRoute{
      spec,
      source};
  };

  std::vector<InputRoute> schema = {
    createRoute("x_source", spec1),
    createRoute("y_source", spec2),
    createRoute("z_source", spec3)};
  // First of all we test if an empty registry behaves as expected, raising a
  // bunch of exceptions.
  InputRecord emptyRecord(schema, {[](size_t) { return nullptr; }, 0});

  BOOST_CHECK_EXCEPTION(emptyRecord.get("x"), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRecord.get("y"), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRecord.get("z"), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRecord.getByPos(0), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRecord.getByPos(1), std::exception, any_exception);
  BOOST_CHECK_EXCEPTION(emptyRecord.getByPos(2), std::exception, any_exception);
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
  InputSpan span{[&inputs](size_t i) { return static_cast<char const*>(inputs[i]); }, inputs.size()};
  InputRecord record{schema, std::move(span)};

  // Checking we can get the whole ref by name
  BOOST_CHECK_NO_THROW(record.get("x"));
  BOOST_CHECK_NO_THROW(record.get("y"));
  BOOST_CHECK_NO_THROW(record.get("z"));
  auto ref00 = record.get("x");
  auto ref10 = record.get("y");
  auto ref20 = record.get("z");
  BOOST_CHECK_EXCEPTION(record.get("err"), std::exception, any_exception);

  // Or we can get it positionally
  BOOST_CHECK_NO_THROW(record.get("x"));
  auto ref01 = record.getByPos(0);
  auto ref11 = record.getByPos(1);
  BOOST_CHECK_EXCEPTION(record.getByPos(10), std::exception, any_exception);

  // This should be exactly the same pointers
  BOOST_CHECK_EQUAL(ref00.header, ref01.header);
  BOOST_CHECK_EQUAL(ref00.payload, ref01.payload);
  BOOST_CHECK_EQUAL(ref10.header, ref11.header);
  BOOST_CHECK_EQUAL(ref10.payload, ref11.payload);

  BOOST_CHECK_EQUAL(record.isValid("x"), true);
  BOOST_CHECK_EQUAL(record.isValid("y"), true);
  BOOST_CHECK_EQUAL(record.isValid("z"), false);

  BOOST_CHECK_EQUAL(record.isValid(0), true);
  BOOST_CHECK_EQUAL(record.isValid(1), true);
  BOOST_CHECK_EQUAL(record.isValid(2), false);
  // This by default is a shortcut for
  //
  // *static_cast<int const *>(record.get("x").payload);
  //
  BOOST_CHECK_EQUAL(record.get<int>("x"), 1);
  BOOST_CHECK_EQUAL(record.get<int>("y"), 2);
  // A few more time just to make sure we are not stateful..
  BOOST_CHECK_EQUAL(record.get<int>("x"), 1);
  BOOST_CHECK_EQUAL(record.get<int>("x"), 1);
}

// TODO:
// - test all `get` implementations
// - create a list of supported types and check that the API compiles
// - test return value optimization for vectors, unique_ptr
// - check objects which work directly on the payload for zero-copy
