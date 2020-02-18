// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_MAPPING_TEST_PARAMETERS_H
#define O2_MCH_MAPPING_TEST_PARAMETERS_H

#include <string>
#include <boost/test/unit_test.hpp>
#include <iostream>
struct TestParameters {

  TestParameters();

  boost::test_tools::assertion_result operator()(boost::unit_test_framework::test_unit_id);

  std::string path;
  bool isTestFileInManuNumbering;
  bool isSegmentationRun3;
};

std::ostream& operator<<(std::ostream& os, const TestParameters& params);

int manu2ds(int deId, int ch);
int ds2manu(int deId, int ch);

#endif
