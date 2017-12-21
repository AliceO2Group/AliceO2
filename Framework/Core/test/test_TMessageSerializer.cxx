// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework AlgorithmSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/TMessageSerializer.h"
#include <boost/test/unit_test.hpp>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestTMessageSerializer) {
  const char* testname = "testname";
  const char* testtitle = "testtitle";
  using namespace o2::framework;

  TObjArray array;
  array.SetOwner();
  array.Add(new TNamed(testname,testtitle));

  FairTMessage msg;
  TMessageSerializer::serialize(msg, &array);

  auto buf = as_span(msg);
  BOOST_CHECK_EQUAL(buf.size(), msg.BufferSize());
  BOOST_CHECK_EQUAL(static_cast<void*>(buf.data()), static_cast<void*>(msg.Buffer()));
  auto out = TMessageSerializer::deserialize(buf);

  TObjArray* outarr = dynamic_cast<TObjArray*>(out.get());
  BOOST_CHECK_EQUAL(out.get(),outarr);
  TNamed* named = dynamic_cast<TNamed*>(outarr->At(0));
  BOOST_CHECK_EQUAL(static_cast<void*>(named),static_cast<void*>(outarr->At(0)));
  BOOST_CHECK_EQUAL(named->GetName(), testname);
  BOOST_CHECK_EQUAL(named->GetTitle(), testtitle);
}
