// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test Framework InputSpec
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include "Framework/InputSpec.h"
#include "Framework/DataSpecUtils.h"
#include "Headers/DataHeader.h"
#include <boost/test/unit_test.hpp>
#include <algorithm>
#include <vector>

using namespace o2::framework;
using namespace o2::framework::data_matcher;

BOOST_AUTO_TEST_CASE(TestSorting)
{
  // At some point
  std::vector<InputSpec> inputs{
    InputSpec{"foo", {"TST", "B"}},
    InputSpec{"bar", {"TST", "A"}}};
  std::swap(inputs[0], inputs[1]);
  auto sorter = [](InputSpec const& a, InputSpec const& b) {
    return a.binding < b.binding;
  };
  std::stable_sort(inputs.begin(), inputs.end(), sorter);
}

BOOST_AUTO_TEST_CASE(TestCreation)
{
  // At some point
  std::vector<InputSpec> inputs{
    InputSpec{"everything", "TST", "B", 0},
    InputSpec{"0-subspec", "TST", "B"},
    InputSpec{"wildcard-subspec", {"TST", "A"}},
    InputSpec{"wildcard-desc-and-subspec", o2::header::DataOrigin{"TST"}},
    InputSpec{"everything-again", {"TST", "B", 0}}};
  std::swap(inputs[0], inputs[1]);
  auto sorter = [](InputSpec const& a, InputSpec const& b) {
    return a.binding < b.binding;
  };
  std::stable_sort(inputs.begin(), inputs.end(), sorter);
}
