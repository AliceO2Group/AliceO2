// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test ValueMonitor
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonUtils/ValueMonitor.h"
#include <TRandom.h>

using namespace o2;

BOOST_AUTO_TEST_CASE(ValueMonitor_test)
{
  utils::ValueMonitor m("foo.root");
  for (int i = 0; i < 100000; ++i) {
    double x = gRandom->Gaus(0, 10) * 10;
    m.Collect<float>("x", (float)x);
    m.Collect<int>("xi", (int)x * 20);
  }
}
