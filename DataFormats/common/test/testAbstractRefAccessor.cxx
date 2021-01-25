// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test AbstractRefAccessor class
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <numeric>
#include <boost/test/unit_test.hpp>
#include "CommonDataFormat/AbstractRefAccessor.h"
#include "CommonDataFormat/AbstractRef.h"
#include "Framework/Logger.h"

using namespace o2::dataformats;

struct Base {
  int b = 0;
};

struct Foo1 : public Base {
  int f1 = 0;
};

struct Foo2 : public Foo1 {
  int f2 = 0;
};

struct Bar1 : public Base {
  double b1 = 0.;
};

struct GloIdx : public AbstractRef<25, 4, 3> {
  enum Source : uint8_t { // provenance of the
    ID0,
    ID1,
    ID2,
    ID3,
    NSources
  };
  using AbstractRef<25, 4, 3>::AbstractRef;
};

// basic AbstractRefAccessor
BOOST_AUTO_TEST_CASE(AbstractRefAccess)
{

  std::vector<Base> vb(10);
  std::vector<Foo1> vf(10);
  std::array<Foo2, 10> af;
  std::vector<Bar1> bar(10);

  std::vector<GloIdx> vid;

  for (int i = 0; i < 10; i++) {
    vb[i].b = GloIdx::ID0 * 100 + i;
    vid.emplace_back(i, GloIdx::ID1);

    vf[i].b = GloIdx::ID1 * 100 + i;
    vf[i].f1 = 0.5 + 100 + i;
    vid.emplace_back(i, GloIdx::ID2);

    af[i].b = GloIdx::ID2 * 100 + i;
    af[i].f2 = 0.8 + 100 + i;
    vid.emplace_back(i, GloIdx::ID3);

    bar[i].b = GloIdx::ID3 * 100 + i;
    bar[i].b1 = 0.2 + 300 + i;
    vid.emplace_back(i, GloIdx::ID3);
  }

  AbstractRefAccessor<Base, GloIdx::NSources> acc;

  acc.registerContainer(vb, GloIdx::ID0);
  acc.registerContainer(vf, GloIdx::ID1);
  acc.registerContainer(af, GloIdx::ID2);
  acc.registerContainer(bar, GloIdx::ID3);

  size_t nid = vid.size();
  for (size_t i = 0; i < nid; i++) {
    auto gid = vid[i];
    const auto& obj = acc.get(gid);
    int expect = gid.getSource() * 100 + i / GloIdx::NSources;
    LOG(INFO) << i << " ? " << obj.b << " == " << expect << " for " << gid.getRaw();
    BOOST_CHECK(obj.b == expect);
  }

  const auto& barEl = acc.get_as<Bar1>(vid.back());
  LOG(INFO) << " ? " << barEl.b1 << " == " << bar.back().b1;
  BOOST_CHECK(barEl.b1 == bar.back().b1);
}
