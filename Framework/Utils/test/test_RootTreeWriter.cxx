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
#include "Utils/MakeRootTreeWriterSpec.h"
#include <vector>
#include <memory>

using namespace o2::framework;

BOOST_AUTO_TEST_CASE(test_RootTreeWriter_static)
{
  // need to mimic a context to actually call the processing
  // for now just test the besic compilation and setup
  RootTreeWriter writer("test.root", "testtree",                                    // file and tree name
                        RootTreeWriter::BranchDef<int>{ "input1", "branchint" },    // branch definition
                        RootTreeWriter::BranchDef<float>{ "input2", "branchfloat" } // branch definition
                        );

  BOOST_CHECK(writer.getStoreSize() == 2);
}

template <typename T>
using BranchDefinition = MakeRootTreeWriterSpec::BranchDefinition<T>;

BOOST_AUTO_TEST_CASE(test_RootTreeWriterSpec)
{
  // setup the spec helper and retrieve the spec by calling the operator
  MakeRootTreeWriterSpec("writer-process",                                                                   //
                         BranchDefinition<int>{ InputSpec{ "input1", "TST", "INTDATA" }, "intbranch" },      //
                         BranchDefinition<float>{ InputSpec{ "input2", "TST", "FLOATDATA" }, "floatbranch" } //
                         )();
}
