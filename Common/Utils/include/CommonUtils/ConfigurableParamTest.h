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

#ifndef COMMON_CONFIGURABLE_PARAM_TEST_H_
#define COMMON_CONFIGURABLE_PARAM_TEST_H_

#include "CommonUtils/ConfigurableParam.h"
#include "CommonUtils/ConfigurableParamHelper.h"

namespace o2::conf::test
{
struct TestParam : public o2::conf::ConfigurableParamHelper<TestParam> {
  enum TestEnum : uint8_t {
    A,
    B,
    C
  };

  int iValue{42};
  float fValue{3.14};
  double dValue{3.14};
  bool bValue{true};
  unsigned uValue{1};
  long lValue{1};
  unsigned long ulValue{1};
  long long llValue{1};
  unsigned long long ullValue{1};
  std::string sValue = "default";
  int iValueProvenanceTest{0};
  TestEnum eValue = TestEnum::C;
  int caValue[3] = {0, 1, 2};

  O2ParamDef(TestParam, "TestParam");
};
} // namespace o2::conf::test

#endif
