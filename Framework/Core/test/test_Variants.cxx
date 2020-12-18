// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework VariantTest
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "Framework/Variant.h"
#include <sstream>
#include <cstring>

using namespace o2::framework;

bool unknown_type(RuntimeErrorRef const& ref)
{
  auto& err = error_from_ref(ref);
  return strcmp(err.what, "Mismatch between types") == 0;
}

BOOST_AUTO_TEST_CASE(VariantTest)
{
  std::ostringstream ss{};
  Variant a(10);
  BOOST_CHECK(a.get<int>() == 10);
  ss << a;
  Variant b(10.1f);
  BOOST_CHECK(b.get<float>() == 10.1f);
  ss << b;
  Variant c(10.2);
  BOOST_CHECK(c.get<double>() == 10.2);
  ss << c;
  BOOST_CHECK_EXCEPTION(a.get<char*>(), RuntimeErrorRef, unknown_type);
  Variant d("foo");
  ss << d;
  BOOST_CHECK(std::string(d.get<const char*>()) == "foo");
  BOOST_CHECK(std::string(d.get<std::string_view>()) == "foo");
  BOOST_CHECK(std::string(d.get<std::string>()) == "foo");

  Variant e(true);
  BOOST_CHECK_EQUAL(e.get<bool>(), true);

  Variant f(false);
  BOOST_CHECK_EQUAL(f.get<bool>(), false);

  BOOST_CHECK(ss.str() == "1010.110.2foo");
  // Spotted valgrind error while deleting a vector of variants.
  std::vector<Variant> vector{1, 1.2, 1.1f, "foo"};
  Variant sa("foo");
  Variant sb(sa);            // Copy constructor
  Variant sc(std::move(sa)); // Move constructor
  Variant sd = sc;           // Copy assignment

  BOOST_CHECK(std::string(sb.get<const char*>()) == "foo");
  BOOST_CHECK(std::string(sc.get<const char*>()) == "foo");
  BOOST_CHECK(std::string(sd.get<const char*>()) == "foo");
  BOOST_CHECK(sa.get<const char*>() == nullptr);

  int iarr[] = {1, 2, 3, 4, 5};
  float farr[] = {0.2, 0.3, 123.123, 123.123, 3.005e-5, 1.1e6};
  std::vector<double> dvec = {0.1, 0.2, 0.4, 0.9, 1.3, 14.5, 123.234, 1.213e-20};
  Variant viarr(iarr, 5);
  Variant vfarr(farr, 6);
  Variant vdvec(dvec);

  BOOST_CHECK(viarr.size() == 5);
  BOOST_CHECK(viarr.get<int*>() != iarr);
  for (auto i = 0u; i < viarr.size(); ++i) {
    BOOST_CHECK(iarr[i] == (viarr.get<int*>())[i]);
  }

  BOOST_CHECK(vfarr.size() == 6);
  BOOST_CHECK(vfarr.get<float*>() != farr);
  for (auto i = 0u; i < vfarr.size(); ++i) {
    BOOST_CHECK(farr[i] == (vfarr.get<float*>())[i]);
  }

  BOOST_CHECK(vdvec.size() == dvec.size());
  BOOST_CHECK(vdvec.get<double*>() != dvec.data());
  for (auto i = 0u; i < dvec.size(); ++i) {
    BOOST_CHECK(dvec[i] == (vdvec.get<double*>())[i]);
  }

  Variant fb(vfarr);            // Copy constructor
  Variant fc(std::move(vfarr)); // Move constructor
  Variant fd = fc;              // Copy assignment

  BOOST_CHECK(vfarr.get<float*>() == nullptr);

  BOOST_CHECK(fb.get<float*>() != farr);
  for (auto i = 0u; i < fb.size(); ++i) {
    BOOST_CHECK(farr[i] == (fb.get<float*>())[i]);
  }
  BOOST_CHECK(fc.get<float*>() != farr);
  for (auto i = 0u; i < fc.size(); ++i) {
    BOOST_CHECK(farr[i] == (fc.get<float*>())[i]);
  }
  BOOST_CHECK(fd.get<float*>() != farr);
  for (auto i = 0u; i < fd.size(); ++i) {
    BOOST_CHECK(farr[i] == (fd.get<float*>())[i]);
  }

  std::vector<std::string> vstrings{"s1", "s2", "s3"};
  std::string strings[] = {"l1", "l2", "l3"};
  Variant vstr(strings, 3);
  Variant vvstr(vstrings);

  BOOST_CHECK(vstr.size() == 3);
  BOOST_CHECK(vvstr.size() == 3);
  for (auto i = 0u; i < vstr.size(); ++i) {
    BOOST_CHECK(strings[i] == (vstr.get<std::string*>())[i]);
  }
  for (auto i = 0u; i < vvstr.size(); ++i) {
    BOOST_CHECK(vstrings[i] == (vvstr.get<std::string*>())[i]);
  }

  Variant vsc(vstr);            // Copy constructor
  Variant vsm(std::move(vstr)); // Move constructor
  Variant vscc = vsm;           // Copy assignment
  for (auto i = 0u; i < vsm.size(); ++i) {
    BOOST_CHECK(strings[i] == (vsm.get<std::string*>())[i]);
  }
  for (auto i = 0u; i < vscc.size(); ++i) {
    BOOST_CHECK(strings[i] == (vscc.get<std::string*>())[i]);
  }
}
