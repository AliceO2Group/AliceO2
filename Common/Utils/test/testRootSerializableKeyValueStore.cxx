// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test RootSerKeyValueStore
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "CommonUtils/RootSerializableKeyValueStore.h"
#include <TMemFile.h>
#include <TH1F.h>
#include <vector>
#include <iostream>

using namespace o2;
using namespace o2::utils;

BOOST_AUTO_TEST_CASE(write_read_test)
{
  RootSerializableKeyValueStore s;

  // put POD some stuff
  double x = 1.1;
  s.put("x", x);
  s.put("i", 110);

  // put some complex classes (need dictionary)
  std::string str = "hello";
  s.put("str", str);

  TH1F h1("th1name", "th1name", 100, 0, 99);
  h1.FillRandom("gaus", 10000);
  s.put("h1", h1);

  // check basic assumption that hash_code is unique (encouraged by standard)
  BOOST_CHECK(std::type_index(typeid(std::vector<double>*)).hash_code() == 10319097832066014690UL);

  // retrieve
  BOOST_CHECK(s.get<std::string>("str")->compare(str) == 0);
  BOOST_CHECK(*(s.get<double>("x")) == x);

  std::cerr << "TESTING FILE IO\n";
  TMemFile f("tmp.root", "RECREATE");
  f.WriteObject(&s, "map");

  RootSerializableKeyValueStore* s2 = nullptr;
  f.GetObject("map", s2);
  BOOST_CHECK(s2 != nullptr);
  if (s2) {
    auto t1 = s2->get<double>("x");
    BOOST_CHECK(t1 != nullptr);
    BOOST_CHECK(t1 && *(t1) == x);

    auto i1 = s2->get<int>("i");
    BOOST_CHECK(i1 != nullptr);
    BOOST_CHECK(i1 && *(i1) == 110);

    auto t2 = s2->get<std::string>("str");
    BOOST_CHECK(t2 && t2->compare(str) == 0);

    auto t3 = s2->get<std::string>("str");
    BOOST_CHECK(t3 && t3->compare(str) == 0);

    auto histo = s2->get<TH1F>("h1");
    BOOST_CHECK(histo);
  }
  f.Close();
}
