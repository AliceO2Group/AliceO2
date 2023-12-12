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

#include <catch_amalgamated.hpp>
#include "Framework/Variant.h"
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

TEST_CASE("VariantTest")
{
  std::ostringstream ss{};
  Variant a(10);
  REQUIRE(a.get<int>() == 10);
  ss << a;
  Variant b(10.1f);
  REQUIRE(b.get<float>() == 10.1f);
  ss << b;
  Variant c(10.2);
  REQUIRE(c.get<double>() == 10.2);
  ss << c;
  REQUIRE_THROWS_AS(a.get<char*>(), RuntimeErrorRef);
  Variant d("foo");
  ss << d;
  REQUIRE(std::string(d.get<const char*>()) == "foo");
  REQUIRE(std::string(d.get<std::string_view>()) == "foo");
  REQUIRE(std::string(d.get<std::string>()) == "foo");

  Variant e(true);
  REQUIRE(e.get<bool>() == true);

  Variant f(false);
  REQUIRE(f.get<bool>() == false);

  REQUIRE(ss.str() == "1010.110.2foo");
  // Spotted valgrind error while deleting a vector of variants.
  std::vector<Variant> vector{1, 1.2, 1.1f, "foo"};
  Variant sa("foo");
  Variant sb(sa);            // Copy constructor
  Variant sc(std::move(sa)); // Move constructor
  Variant sd = sc;           // Copy assignment

  REQUIRE(std::string(sb.get<const char*>()) == "foo");
  REQUIRE(std::string(sc.get<const char*>()) == "foo");
  REQUIRE(std::string(sd.get<const char*>()) == "foo");
  REQUIRE(sa.get<const char*>() == nullptr);

  int iarr[] = {1, 2, 3, 4, 5};
  float farr[] = {0.2, 0.3, 123.123, 123.123, 3.005e-5, 1.1e6};
  std::vector<double> dvec = {0.1, 0.2, 0.4, 0.9, 1.3, 14.5, 123.234, 1.213e-20};
  Variant viarr(iarr, 5);
  Variant vfarr(farr, 6);
  Variant vdvec(dvec);

  REQUIRE(viarr.size() == 5);
  REQUIRE(viarr.get<int*>() != iarr);
  for (auto i = 0u; i < viarr.size(); ++i) {
    REQUIRE(iarr[i] == (viarr.get<int*>())[i]);
  }

  REQUIRE(vfarr.size() == 6);
  REQUIRE(vfarr.get<float*>() != farr);
  for (auto i = 0u; i < vfarr.size(); ++i) {
    REQUIRE(farr[i] == (vfarr.get<float*>())[i]);
  }

  REQUIRE(vdvec.size() == dvec.size());
  REQUIRE(vdvec.get<double*>() != dvec.data());
  for (auto i = 0u; i < dvec.size(); ++i) {
    REQUIRE(dvec[i] == (vdvec.get<double*>())[i]);
  }

  Variant fb(vfarr);            // Copy constructor
  Variant fc(std::move(vfarr)); // Move constructor
  Variant fd = fc;              // Copy assignment

  REQUIRE(vfarr.get<float*>() == nullptr);

  REQUIRE(fb.get<float*>() != farr);
  for (auto i = 0u; i < fb.size(); ++i) {
    REQUIRE(farr[i] == (fb.get<float*>())[i]);
  }
  REQUIRE(fc.get<float*>() != farr);
  for (auto i = 0u; i < fc.size(); ++i) {
    REQUIRE(farr[i] == (fc.get<float*>())[i]);
  }
  REQUIRE(fd.get<float*>() != farr);
  for (auto i = 0u; i < fd.size(); ++i) {
    REQUIRE(farr[i] == (fd.get<float*>())[i]);
  }

  std::vector<std::string> vstrings{"s1", "s2", "s3"};
  std::string strings[] = {"l1", "l2", "l3"};
  Variant vstr(strings, 3);
  Variant vvstr(vstrings);

  REQUIRE(vstr.size() == 3);
  REQUIRE(vvstr.size() == 3);
  for (auto i = 0u; i < vstr.size(); ++i) {
    REQUIRE(strings[i] == (vstr.get<std::string*>())[i]);
  }
  for (auto i = 0u; i < vvstr.size(); ++i) {
    REQUIRE(vstrings[i] == (vvstr.get<std::string*>())[i]);
  }

  Variant vsc(vstr);            // Copy constructor
  Variant vsm(std::move(vstr)); // Move constructor
  Variant vscc = vsm;           // Copy assignment
  for (auto i = 0u; i < vsm.size(); ++i) {
    REQUIRE(strings[i] == (vsm.get<std::string*>())[i]);
  }
  for (auto i = 0u; i < vscc.size(); ++i) {
    REQUIRE(strings[i] == (vscc.get<std::string*>())[i]);
  }

  Variant vsca(vvstr);            // Copy constructor
  Variant vsma(std::move(vvstr)); // Move constructor
  Variant vscca = vsma;           // Copy assignment
  for (auto i = 0u; i < vsma.size(); ++i) {
    REQUIRE(vstrings[i] == (vsma.get<std::string*>())[i]);
  }
  for (auto i = 0u; i < vscca.size(); ++i) {
    REQUIRE(vstrings[i] == (vscca.get<std::string*>())[i]);
  }

  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  Variant vmm(mm);
  auto const& mmc = vmm.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(mmc(i, j) == mm(i, j));
    }
  }

  Variant vmmc(vmm);            // Copy constructor
  Variant vmmm(std::move(vmm)); // Move constructor
  Variant vmma = vmmm;          // Copy assignment
  auto const& mmc2 = vmmc.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(mmc2(i, j) == mm(i, j));
    }
  }
  auto const& mmc3 = vmma.get<Array2D<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(mmc3(i, j) == mm(i, j));
    }
  }
  std::stringstream ssm;
  ssm << vmma;
  REQUIRE(ssm.str() == "{\"values\":[[0.10000000149011612,0.20000000298023225,0.30000001192092898,0.4000000059604645],[0.5,0.6000000238418579,0.699999988079071,0.800000011920929],[0.8999999761581421,1.0,1.100000023841858,1.2000000476837159]]}");

  LabeledArray<float> laf{&m[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  Variant vlaf(laf);
  auto const& lafc = vlaf.get<LabeledArray<float>>();
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(laf.get(i, j) == lafc.get(i, j));
    }
  }

  Variant vlafc(vlaf);            // Copy constructor
  Variant vlafm(std::move(vlaf)); // Move constructor
  Variant vlafa = vlafm;          // Copy assignment
  auto const& lafc2 = vlafc.get<LabeledArray<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      REQUIRE(lafc2.get(i, j) == mm(i, j));
    }
  }
  auto const& lafc3 = vlafa.get<LabeledArray<float>>();
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      REQUIRE(lafc3.get(i, j) == mm(i, j));
    }
  }

  std::vector<Variant> collection;
  collection.push_back(vlafc);
  collection.push_back(vlafm);
  collection.push_back(vlafa);
}

TEST_CASE("Array2DTest")
{
  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  Array2D mm(&m[0][0], 3, 4);
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      REQUIRE(mm(i, j) == m[i][j]);
    }
  }
  std::vector<float> v = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2};
  Array2D mv(v, 3, 4);
  for (auto i = 0U; i < 3; ++i) {
    for (auto j = 0U; j < 4; ++j) {
      REQUIRE(mm(i, j) == v[i * 4 + j]);
    }
  }
  for (auto i = 0U; i < 3; ++i) {
    auto const& vv = mm[i];
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(vv[j] == mm(i, j));
    }
  }
  std::vector<std::string> s = {"one", "two", "three", "four"};
  Array2D ms(s, 4, 1);
  for (auto i = 0U; i < 4; ++i) {
    REQUIRE(ms(i, 0) == s[i]);
  }
}

TEST_CASE("LabeledArrayTest")
{
  float m[3][4] = {{0.1, 0.2, 0.3, 0.4}, {0.5, 0.6, 0.7, 0.8}, {0.9, 1.0, 1.1, 1.2}};
  std::string mS[3][4] = {{"a", "b", "c", "d"}, {"e", "f", "g", "h"}, {"i", "l", "m", "n"}};
  std::string xl[] = {"c1", "c2", "c3", "c4"};
  std::string yl[] = {"r1", "r2", "r3"};
  LabeledArray<float> laf{&m[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  LabeledArray<std::string> las{&mS[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  for (auto i = 0u; i < 3; ++i) {
    for (auto j = 0u; j < 4; ++j) {
      REQUIRE(laf.get(yl[i].c_str(), xl[j].c_str()) == laf.get(i, j));
      REQUIRE(laf.get(i, xl[j].c_str()) == laf.get(i, j));
      REQUIRE(laf.get(yl[i].c_str(), j) == laf.get(i, j));

      REQUIRE(las.get(yl[i].c_str(), xl[j].c_str()) == las.get(i, j));
      REQUIRE(las.get(i, xl[j].c_str()) == las.get(i, j));
      REQUIRE(las.get(yl[i].c_str(), j) == las.get(i, j));
    }
  }
}

TEST_CASE("VariantTreeConversionsTest")
{
  std::vector<std::string> vstrings{"0 1", "0 2", "0 3"};
  Variant vvstr(std::move(vstrings));

  auto tree = vectorToBranch(vvstr.get<std::string*>(), vvstr.size());
  auto v = Variant(vectorFromBranch<std::string>(tree));

  for (auto i = 0U; i < vvstr.size(); ++i) {
    REQUIRE(vvstr.get<std::string*>()[i] == v.get<std::string*>()[i]);
  }
}

TEST_CASE("VariantJSONConversionsTest")
{
  int iarr[] = {1, 2, 3, 4, 5};
  Variant viarr(iarr, 5);
  std::stringstream os;
  VariantJSONHelpers::write(os, viarr);

  std::stringstream is;
  is.str(os.str());
  auto v = VariantJSONHelpers::read<VariantType::ArrayInt>(is);
  for (auto i = 0u; i < viarr.size(); ++i) {
    REQUIRE(v.get<int*>()[i] == viarr.get<int*>()[i]);
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
      REQUIRE(vmm.get<Array2D<float>>()(i, j) == vm.get<Array2D<float>>()(i, j));
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
      REQUIRE(vlaf.get<LabeledArray<float>>().get(i, j) == vlafc.get<LabeledArray<float>>().get(i, j));
    }
  }

  std::string mS[3][4] = {{"a", "b", "c", "d"}, {"e", "f", "g", "h"}, {"i", "l", "m", "n"}};
  LabeledArray<std::string> las{&mS[0][0], 3, 4, {"r1", "r2", "r3"}, {"c1", "c2", "c3", "c4"}};
  Variant vms(las);
  std::stringstream ossl;
  VariantJSONHelpers::write(ossl, vms);

  std::stringstream issl;
  issl.str(ossl.str());
  auto vmsa = VariantJSONHelpers::read<VariantType::LabeledArrayString>(issl);

  for (auto i = 0U; i < vmsa.get<LabeledArray<std::string>>().rows(); ++i) {
    for (auto j = 0U; j < vmsa.get<LabeledArray<std::string>>().cols(); ++j) {
      REQUIRE(vmsa.get<LabeledArray<std::string>>().get(i, j) == vms.get<LabeledArray<std::string>>().get(i, j));
    }
  }

  std::vector<std::string> vstrings{"myoption_one", "myoption_two"};
  Variant vvstr(vstrings);
  std::stringstream osal;
  VariantJSONHelpers::write(osal, vvstr);

  std::stringstream isal;
  isal.str(osal.str());
  auto vvstra = VariantJSONHelpers::read<VariantType::ArrayString>(isal);

  for (auto i = 0U; i < vvstra.size(); ++i) {
    REQUIRE(vstrings[i] == vvstra.get<std::string*>()[i]);
  }
}
