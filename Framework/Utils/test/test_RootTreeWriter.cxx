// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework Utils RootTreeWriter
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include "Utils/RootTreeWriter.h"
#include <vector>
#include <memory>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(test_RootTreeWriter_static)
{
  // need to mimic a context to actually call the processing
  // for now just test the besic compilation and setup
  using WriterT = RootTreeWriter<int, float>;
  WriterT writer("test.root", "testtree", // file and tree name
                 "input1", "branchint",   // branch config pair
                 "input2", "branchfloat"  // branch config pair
                 );

  BOOST_CHECK(writer.store_size == 2);
  BOOST_CHECK((std::is_same<typename WriterT::element<0>::type, int>::value == true));
  BOOST_CHECK((std::is_same<typename WriterT::element<1>::type, float>::value == true));
}

BOOST_AUTO_TEST_CASE(test_RootTreeWriter_runtime)
{
  // use the writer with runtime init from a vector of config pairs
  using WriterT = RootTreeWriter<int, float>;
  std::vector<std::pair<std::string, std::string>> branchConfig;

  // exception must be raised because of incomplete configuration
  branchConfig.emplace_back("input1", "branchint");
  auto createWriterFct = [&branchConfig]() { return std::make_unique<WriterT>("test.root", "tree", branchConfig); };
  BOOST_CHECK_THROW(createWriterFct(), std::runtime_error);

  // test with correct configuration
  branchConfig.emplace_back("input2", "branchfloat");
  auto writer = createWriterFct();
}
