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

#include "TestParameters.h"
#include <array>

TestParameters::TestParameters() : path(""), isTestFileInManuNumbering{false}, isSegmentationRun3{true}
{
  auto& ts = boost::unit_test::framework::master_test_suite();
  auto n = ts.argc;
  if (n >= 2) {
    std::string testpos{"--testpos"};
    std::string testnumbering{"--manunumbering"};
    std::string testrun2{"--run2"};
    for (auto i = 0; i < n; i++) {
      if (testpos == ts.argv[i] && i < n - 1) {
        path = ts.argv[i + 1];
        ++i;
      }
      if (testnumbering == ts.argv[i]) {
        isTestFileInManuNumbering = true;
      }
      if (testrun2 == ts.argv[i]) {
        isSegmentationRun3 = false;
      }
    }
  }
}

boost::test_tools::assertion_result TestParameters::operator()(boost::unit_test_framework::test_unit_id)
{
  return !path.empty();
}

std::ostream& operator<<(std::ostream& os, const TestParameters& params)
{
  os << "path:" << params.path
     << " manu:" << params.isTestFileInManuNumbering
     << " run3:" << params.isSegmentationRun3;
  return os;
}

std::array<int, 64> refManu2ds_st345 = {
  63, 62, 61, 60, 59, 57, 56, 53, 51, 50, 47, 45, 44, 41, 38, 35,
  36, 33, 34, 37, 32, 39, 40, 42, 43, 46, 48, 49, 52, 54, 55, 58,
  7, 8, 5, 2, 6, 1, 3, 0, 4, 9, 10, 15, 17, 18, 22, 25,
  31, 30, 29, 28, 27, 26, 24, 23, 20, 21, 16, 19, 12, 14, 11, 13};

std::array<int, 64> refManu2ds_st12 = {
  36, 35, 34, 33, 32, 37, 38, 43, 45, 47, 49, 50, 53, 41, 39, 40,
  63, 62, 61, 60, 59, 58, 56, 57, 54, 55, 52, 51, 48, 46, 44, 42,
  31, 30, 29, 28, 27, 26, 25, 24, 22, 23, 20, 18, 17, 15, 13, 11,
  4, 3, 2, 1, 0, 5, 6, 10, 12, 14, 16, 19, 21, 8, 7, 9};

int manu2ds(int deId, int ch)
{
  if (deId < 500) {
    return refManu2ds_st12[ch];
  }
  return refManu2ds_st345[ch];
}

std::array<int, 64> reverse(std::array<int, 64> a)
{
  std::array<int, 64> r;
  for (auto i = 0; i < a.size(); i++) {
    r[a[i]] = i;
  }
  return r;
}

int ds2manu(int deId, int ch)
{
  static std::array<int, 64> ds2manu_st12 = reverse(refManu2ds_st12);
  static std::array<int, 64> ds2manu_st345 = reverse(refManu2ds_st345);
  if (deId < 500) {
    return ds2manu_st12[ch];
  }
  return ds2manu_st345[ch];
}
