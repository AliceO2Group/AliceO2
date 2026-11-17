// Copyright 2019-2026 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testGetOrMakeBranch.cxx
/// \brief Checks that getOrMakeBranch leaves a valid address on an existing branch

#define BOOST_TEST_MODULE Test GetOrMakeBranch
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>

#include "DetectorsBase/Detector.h"
#include <TTree.h>
#include <cstring>
#include <vector>

namespace
{
// attach the data in their own stack frame, which is gone when the branch is filled
__attribute__((noinline)) TBranch* attach(TTree& tree, std::vector<int>* data)
{
  return o2::base::getOrMakeBranch(tree, "data", data);
}

__attribute__((noinline)) TBranch* attachPointer(TTree& tree, std::vector<int>** data)
{
  return o2::base::getOrMakeBranch(tree, "pointer", data);
}

// overwrites the stack below the caller
__attribute__((noinline)) void clobberStack()
{
  volatile char buffer[4096];
  std::memset(const_cast<char*>(buffer), 0xab, sizeof(buffer));
}
} // namespace

BOOST_AUTO_TEST_CASE(FillExistingBranch)
{
  TTree tree("t", "t");
  tree.SetDirectory(nullptr);
  std::vector<std::vector<int>> entries{{1, 2, 3}, {4, 5}, {6}};
  for (auto& entry : entries) {
    auto pointer = &entry;
    auto br = attach(tree, &entry);
    auto brPointer = attachPointer(tree, &pointer);
    clobberStack();
    br->Fill();
    brPointer->Fill();
    br->ResetAddress();
    brPointer->ResetAddress();
  }
  tree.SetEntries(entries.size());

  std::vector<int>* read = nullptr;
  std::vector<int>* readPointer = nullptr;
  tree.SetBranchAddress("data", &read);
  tree.SetBranchAddress("pointer", &readPointer);
  BOOST_REQUIRE_EQUAL(tree.GetEntries(), entries.size());
  for (size_t i = 0; i < entries.size(); ++i) {
    tree.GetEntry(i);
    BOOST_CHECK(*read == entries[i]);
    BOOST_CHECK(*readPointer == entries[i]);
  }
}
