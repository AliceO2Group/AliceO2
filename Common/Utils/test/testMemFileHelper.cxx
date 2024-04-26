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

/// @file   testMemFileHelper.cxx
/// @author ruben.shahoyan@cern.ch
/// @brief  unit tests for TMemFile and its binary image creation

#define BOOST_TEST_MODULE MemFileUtils unit test
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <filesystem>
#include <cstdio>
#include <vector>
#include <TFile.h>
#include "CommonUtils/MemFileHelper.h"

BOOST_AUTO_TEST_CASE(test_memfile_helper)
{
  std::vector<int> vec = {1, 2, 3};
  std::string fname = "test_MemFile.root";
  std::string objname = o2::utils::MemFileHelper::getClassName(vec);
  BOOST_CHECK(!objname.empty());
  auto img = o2::utils::MemFileHelper::createFileImage(vec, "test_MemFile.root");
  FILE* fp = fopen(fname.c_str(), "wb");
  fwrite(img->data(), img->size(), 1, fp);
  fclose(fp);
  //
  // try to read back
  TFile rdf(fname.c_str());
  auto* rvec = (std::vector<int>*)rdf.GetObjectChecked(objname.c_str(), objname.c_str());
  BOOST_CHECK(rvec);
  BOOST_CHECK(*rvec == vec);
}

#include "CommonUtils/FileSystemUtils.h"
#include <cstdlib>

BOOST_AUTO_TEST_CASE(test_expandenv)
{
  {
    std::string noenv("simple_file.root");
    auto expandedFileName = o2::utils::expandShellVarsInFileName(noenv);
    BOOST_CHECK(expandedFileName.size() > 0);
    BOOST_CHECK_EQUAL(expandedFileName, noenv);
  }

  {
    std::string withenv("${PWD}/simple_file.root");
    auto expandedFileName = o2::utils::expandShellVarsInFileName(withenv);
    BOOST_CHECK(expandedFileName.size() > 0);
  }

  {
    setenv("FOO_123", "BAR", 0);
    std::string withenv("/tmp/${FOO_123}/simple_file.root");
    auto expandedFileName = o2::utils::expandShellVarsInFileName(withenv);
    BOOST_CHECK_EQUAL(expandedFileName, "/tmp/BAR/simple_file.root");
  }

  { // what if the variable doesn't exist --> should return unmodified string
    std::string withenv("/tmp/${FOO_DOESNOTEXIST}/simple_file.root");
    auto expandedFileName = o2::utils::expandShellVarsInFileName(withenv);
    BOOST_CHECK_EQUAL(expandedFileName, withenv);
  }
}