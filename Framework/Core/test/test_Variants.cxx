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
#include "Framework/VariantStringHelpers.h"
#include "Framework/VariantPropertyTreeHelpers.h"
#include "Framework/VariantJSONHelpers.h"
#include <sstream>
#include <cstring>

using namespace o2::framework;

bool unknown_type(RuntimeErrorRef const& ref)
{
  auto const& err = error_from_ref(ref);
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

  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  Variant vmm(mm);
  auto const& mmc = vmm.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(mmc(i, j) == mm(i, j));
    }
  }

  Variant vmmc(vmm);            // Copy constructor
  Variant vmmm(std::move(vmm)); // Move constructor
  Variant vmma = vmmm;          // Copy assignment
  auto const& mmc2 = vmmc.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(mmc2(i, j) == mm(i, j));
    }
  }
  auto const& mmc3 = vmma.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(mmc3(i, j) == mm(i, j));
    }
  }
  std::stringstream ssm;
  ssm << vmma;
  BOOST_CHECK(ssm.str() == "f[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8], [0.9, 1, 1.1, 1.2]]");

  LabeledArray<float> laf{&m[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  Variant vlaf(laf);
  auto const& lafc = vlaf.get<LabeledArray<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(laf.get(i, j) == lafc.get(i, j));
    }
  }

  Variant vlafc(vlaf);            // Copy constructor
  Variant vlafm(std::move(vlaf)); // Move constructor
  Variant vlafa = vlafm;          // Copy assignment
  auto const& lafc2 = vlafc.get<LabeledArray<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(lafc2.get(i, j) == mm(i, j));
    }
  }
  auto const& lafc3 = vlafa.get<LabeledArray<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      BOOST_CHECK(lafc3.get(i, j) == mm(i, j));
    }
  }

  std::vector<Variant> collection;
  collection.push_back(vlafc);
  collection.push_back(vlafm);
  collection.push_back(vlafa);
}

BOOST_AUTO_TEST_CASE(Array2DTest)
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

BOOST_AUTO_TEST_CASE(LabeledArrayTest)
{
  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  std::string xl[] = {"c1", "c2", "c3", "c4"};
  std::string yl[] = {"r1", "r2", "r3"};
  LabeledArray<float> laf{&m[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      BOOST_CHECK(laf.get(yl[i], xl[j]) == laf.get(i, j));
    }
  }
}

BOOST_AUTO_TEST_CASE(VariantConversionsTest)
{
  int iarr[] = {1, 2, 3, 4, 5};
  Variant viarr(iarr, 5);
  std::stringstream os;
  VariantJSONHelpers::write(os, viarr);

  std::stringstream is;
  is.str(os.str());
  auto v = VariantJSONHelpers::read<VariantType::ArrayInt>(is);
  for (auto i = 0u; i < viarr.size(); ++i) {
    BOOST_CHECK_EQUAL(v.get<int*>()[i], viarr.get<int*>()[i]);
  }
  os.str("");

  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  Variant vmm(mm);
  std::stringstream osm;
  VariantJSONHelpers::write(osm, vmm);

  std::stringstream ism;
  ism.str(osm.str());
  auto vm = VariantJSONHelpers::read<VariantType::Array2DFloat>(ism);

  for (auto i = 0u; i < mm.rows; ++i) {
    for (auto j = 0u; j < mm.cols; ++j) {
      BOOST_CHECK_EQUAL(vmm.get<Array2D<float>>()(i, j), vm.get<Array2D<float>>()(i, j));
    }
  }

  LabeledArray<float> laf{&m[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  Variant vlaf(laf);
  std::stringstream osl;
  VariantJSONHelpers::write(osl, vlaf);

  std::stringstream isl;
  isl.str(osl.str());
  auto vlafc = VariantJSONHelpers::read<VariantType::LabeledArrayFloat>(isl);

  for (auto i = 0u; i < vlafc.get<LabeledArray<float>>().rows(); ++i) {
    for (auto j = 0u; j < vlafc.get<LabeledArray<float>>().cols(); ++j) {
      BOOST_CHECK_EQUAL(vlaf.get<LabeledArray<float>>().get(i, j), vlafc.get<LabeledArray<float>>().get(i, j));
    }
  }
}
