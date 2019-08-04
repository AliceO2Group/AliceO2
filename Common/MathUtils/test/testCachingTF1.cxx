// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test CachingTF1
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "MathUtils/CachingTF1.h"
#include <TFile.h>

using namespace o2::base;

BOOST_AUTO_TEST_CASE(CachingTF1_test)
{
  std::string s("std::pow(x, 1.2)*std::exp(-x/3.)");
  CachingTF1 func("testfunction", s.c_str(), 0, 100.);
  const int kNPoints = 500;
  func.SetNpx(kNPoints);
  BOOST_CHECK(func.getIntegralVector().size() == 0);

  BOOST_CHECK(func.GetRandom() > 0);
  auto f = TFile::Open("tmpTF1Cache.root", "recreate");
  BOOST_CHECK(f);
  f->WriteTObject(&func, "func");
  f->Close();
  // open for reading and verify that integral was cached
  f = TFile::Open("tmpTF1Cache.root");
  BOOST_CHECK(f);
  volatile auto func2 = (CachingTF1*)f->Get("func");
  BOOST_CHECK(func2);
  BOOST_CHECK(func2->getIntegralVector().size() == kNPoints + 1);

  // check reference
  auto& ref = *func2;
  BOOST_CHECK(ref.getIntegralVector().size() == kNPoints + 1);
}
