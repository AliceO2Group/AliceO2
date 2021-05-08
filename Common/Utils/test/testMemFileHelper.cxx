// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
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
