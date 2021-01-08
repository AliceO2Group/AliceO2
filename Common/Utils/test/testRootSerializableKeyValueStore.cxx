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
#include <TClass.h>

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

  // this should fail compiling:
  // const char* text = "foo";
  // s.put("cstr", text);

  TH1F h1("th1name", "th1name", 100, 0, 99);
  h1.FillRandom("gaus", 10000);
  s.put("h1", h1);

  // check basic assumption that typeinfo name is unique for basic types
  BOOST_CHECK(strcmp(std::type_index(typeid(double*)).name(), "Pd") == 0);
  BOOST_CHECK(strcmp(std::type_index(typeid(int)).name(), "i") == 0);
  BOOST_CHECK(strcmp(std::type_index(typeid(unsigned int)).name(), "j") == 0);
  BOOST_CHECK(strcmp(std::type_index(typeid(char)).name(), "c") == 0);
  BOOST_CHECK(strcmp(std::type_index(typeid(char*)).name(), "Pc") == 0);
  BOOST_CHECK(strcmp(std::type_index(typeid(unsigned char)).name(), "h") == 0);

  // check assumption that for more complicated types the TClass name is unique
  // (the std::type_index is not standardized)
  BOOST_CHECK(strcmp(TClass::GetClass(typeid(std::vector<double>))->GetName(), "vector<double>") == 0);

  // retrieve
  BOOST_CHECK(s.get<std::string>("str")->compare(str) == 0);
  BOOST_CHECK(*(s.get<double>("x")) == x);
  BOOST_CHECK(s.has("x"));
  BOOST_CHECK(!s.has("x_does_not_exist"));

  // retrieve with state/error information
  using ErrorState = RootSerializableKeyValueStore::GetState;
  ErrorState state;
  {
    auto r1 = s.get<std::string>("str", state);
    BOOST_CHECK(state == ErrorState::kOK);
    auto returnedstring = s.getRef<std::string>("str", state);
    BOOST_CHECK(state == ErrorState::kOK);
    BOOST_CHECK(returnedstring.compare(str) == 0);

    auto r2 = s.get<int>("str", state);
    BOOST_CHECK(r2 == nullptr);
    BOOST_CHECK(state == ErrorState::kWRONGTYPE);
    auto r3 = s.get<int>("str2", state);
    BOOST_CHECK(state == ErrorState::kNOSUCHKEY);
    BOOST_CHECK(r3 == nullptr);

    auto r4 = s.get<TH1F>("non-existend-histogram", state);
    BOOST_CHECK(state == ErrorState::kNOSUCHKEY);
  }

  // put something twice
  s.put<int>("twice", 10);
  s.put<int>("twice", 7);
  BOOST_CHECK(*(s.get<int>("twice")) == 7);

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

    s2->print();
  }
  f.Close();
}
