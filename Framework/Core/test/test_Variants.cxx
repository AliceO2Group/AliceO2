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
  auto const& err = error_from_ref(ref);
  return strcmp(err.what, "Mismatch between types") == 0;
}

BOOST_AUTO_TEST_CASE(MatrixTest)
{
  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(mm(i, j) == m[i][j]);
    }
  }
  std::vector<float> v = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  Array2D mv(v, 3, 4);
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(mm(i, j) == v[i * 4 + j]);
    }
  }
  for (auto i = 0U; i < 3; ++i) {
    auto const& vv = mm[i];
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(vv[j] == mm(i, j));
    }
  }
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
  for (auto i = 0U; i < viarr.size(); ++i) {
    BOOST_CHECK(iarr[i] == (viarr.get<int*>())[i]);
  }

  BOOST_CHECK(vfarr.size() == 6);
  BOOST_CHECK(vfarr.get<float*>() != farr);
  for (auto i = 0U; i < vfarr.size(); ++i) {
    BOOST_CHECK(farr[i] == (vfarr.get<float*>())[i]);
  }

  BOOST_CHECK(vdvec.size() == dvec.size());
  BOOST_CHECK(vdvec.get<double*>() != dvec.data());
  for (auto i = 0U; i < dvec.size(); ++i) {
    BOOST_CHECK(dvec[i] == (vdvec.get<double*>())[i]);
  }

  Variant fb(vfarr);            // Copy constructor
  Variant fc(std::move(vfarr)); // Move constructor
  Variant fd = fc;              // Copy assignment

  BOOST_CHECK(vfarr.get<float*>() == nullptr);

  BOOST_CHECK(fb.get<float*>() != farr);
  for (auto i = 0U; i < fb.size(); ++i) {
    BOOST_CHECK(farr[i] == (fb.get<float*>())[i]);
  }
  BOOST_CHECK(fc.get<float*>() != farr);
  for (auto i = 0U; i < fc.size(); ++i) {
    BOOST_CHECK(farr[i] == (fc.get<float*>())[i]);
  }
  BOOST_CHECK(fd.get<float*>() != farr);
  for (auto i = 0U; i < fd.size(); ++i) {
    BOOST_CHECK(farr[i] == (fd.get<float*>())[i]);
  }

  std::vector<std::string> vstrings{"s1", "s2", "s3"};
  std::string strings[] = {"l1", "l2", "l3"};
  Variant vstr(strings, 3);
  Variant vvstr(vstrings);

  BOOST_CHECK(vstr.size() == 3);
  BOOST_CHECK(vvstr.size() == 3);
  for (auto i = 0U; i < vstr.size(); ++i) {
    BOOST_CHECK(strings[i] == (vstr.get<std::string*>())[i]);
  }
  for (auto i = 0U; i < vvstr.size(); ++i) {
    BOOST_CHECK(vstrings[i] == (vvstr.get<std::string*>())[i]);
  }

  Variant vsc(vstr);            // Copy constructor
  Variant vsm(std::move(vstr)); // Move constructor
  Variant vscc = vsm;           // Copy assignment
  for (auto i = 0U; i < vsm.size(); ++i) {
    BOOST_CHECK(strings[i] == (vsm.get<std::string*>())[i]);
  }
  for (auto i = 0U; i < vscc.size(); ++i) {
    BOOST_CHECK(strings[i] == (vscc.get<std::string*>())[i]);
  }

  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  Variant vmm(mm);
  auto const& mmc = vmm.get<Array2D<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(mmc(i, j) == mm(i, j));
    }
  }

  Variant vmmc(vmm);            // Copy constructor
  Variant vmmm(std::move(vmm)); // Move constructor
  Variant vmma = vmmm;          // Copy assignment
  auto const& mmc2 = vmmc.get<Array2D<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(mmc2(i, j) == mm(i, j));
    }
  }
  auto const& mmc3 = vmma.get<Array2D<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(mmc3(i, j) == mm(i, j));
    }
  }
  std::stringstream ssm;
  ssm << vmma;
  BOOST_CHECK(ssm.str() == "f[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1, 1.1, 1.2]]");
}
