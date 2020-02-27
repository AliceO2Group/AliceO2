// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test MCHRaw CRUEncoder
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "MCHRawElecMap/Mapper.h"
#include <fmt/format.h>
#include <set>
#include <boost/mpl/list.hpp>
#include <gsl/span>
#include <array>
#include "dslist.h"

using namespace o2::mch::raw;

typedef boost::mpl::list<o2::mch::raw::ElectronicMapperDummy,
                         o2::mch::raw::ElectronicMapperGenerated>
  testTypes;

typedef boost::mpl::list<o2::mch::raw::ElectronicMapperGenerated>
  realTypes;

BOOST_AUTO_TEST_SUITE(o2_mch_raw)

BOOST_AUTO_TEST_SUITE(electronicmapperdummy)

// BOOST_AUTO_TEST_CASE(dumpseg)
// {
//   std::map<int, std::vector<int>> dsids;
//
//   o2::mch::mapping::forEachDetectionElement([&dsids](int deId) {
//     auto seg = o2::mch::mapping::segmentation(deId);
//     seg.forEachDualSampa([&dsids, deId](int dsid) {
//       dsids[deId].emplace_back(dsid);
//     });
//   });
//
//   for (auto p : dsids) {
//     std::cout << "dsids[" << p.first << "]={";
//     for (auto i = 0; i < p.second.size(); i++) {
//       std::cout << p.second[i];
//       if (i != p.second.size() - 1) {
//         std::cout << ",";
//       }
//     }
//     std::cout << "};\n";
//   }
//   std::cout << "\n";
// }

auto dslist = createDualSampaMapper();

template <size_t N>
std::set<int> nofDualSampas(std::array<int, N> deIds)
{
  std::set<int> ds;

  for (auto deId : deIds) {
    for (auto dsid : dslist(deId)) {
      ds.insert(encode(DsDetId{deId, dsid}));
    }
  }
  return ds;
}

template <typename T>
std::set<int> nofDualSampasFromMapper(gsl::span<int> deids)
{
  std::set<int> ds;

  auto d2e = o2::mch::raw::createDet2ElecMapper<T>(deids);

  for (auto deid : deids) {
    size_t nref = dslist(deid).size();
    std::set<int> dsForOneDE;
    for (auto dsid : dslist(deid)) {
      DsDetId id{static_cast<uint16_t>(deid), static_cast<uint16_t>(dsid)};
      auto dsel = d2e(id);
      if (dsel.has_value()) {
        auto code = o2::mch::raw::encode(id); // encode to be sure we're counting unique pairs (deid,dsid)
        ds.insert(code);
        dsForOneDE.insert(code);
      }
    };
    if (dsForOneDE.size() != nref) {
      std::cout << fmt::format("ERROR : mapper has {:4d} DS while DE {:4d} should have {:4d}\n", dsForOneDE.size(), deid, nref);
    }
  }
  return ds;
}
template <typename T>
std::set<uint16_t> getSolarUIDs(int deid)
{
  auto d2e = o2::mch::raw::createDet2ElecMapper<T>();
  std::set<uint16_t> solarsForDE;
  for (auto dsid : dslist(deid)) {
    DsDetId id{static_cast<uint16_t>(deid), static_cast<uint16_t>(dsid)};
    auto dsel = d2e(id);
    if (dsel.has_value()) {
      solarsForDE.insert(dsel->solarId());
    }
  }
  return solarsForDE;
}

template <typename T>
std::set<uint16_t> getSolarUIDs()
{
  std::set<uint16_t> solarUIDs;

  for (auto deid : deIdsForAllMCH) {
    std::set<uint16_t> solarsForDE = getSolarUIDs<T>(deid);
    for (auto s : solarsForDE) {
      solarUIDs.insert(s);
    }
  }
  return solarUIDs;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH5R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH5R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH5R);
  BOOST_CHECK_EQUAL(check.size(), expected.size());
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH5L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH5L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH5L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH6R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH6R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH6R);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

// BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
// BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH6L, T, testTypes)
// {
//   auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH6L);
//   auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH6L);
//   BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
// }
//
BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH7R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH7R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH7R);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH7L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH7L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH7L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckNumberOfSolarsPerDetectionElement, T, realTypes)
{
  // Chamber 1
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(100).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(102).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(101).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(103).size(), 0);

  // Chamber 2
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(200).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(202).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(201).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(203).size(), 0);

  // Chamber 3
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(300).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(302).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(301).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(303).size(), 0);

  // Chamber 4
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(400).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(402).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(401).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(403).size(), 0);

  // Chamber 5
  // 5R = 5I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(500).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(501).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(502).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(503).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(504).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(514).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(515).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(516).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(517).size(), 4);
  // 5L = 5O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(505).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(506).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(507).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(508).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(509).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(510).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(511).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(512).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(513).size(), 1);

  // Chamber 6
  // 6R = 6I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(600).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(601).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(602).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(603).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(604).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(614).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(615).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(616).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(617).size(), 4);
  // 6L = 6O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(605).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(606).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(607).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(608).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(609).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(610).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(611).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(612).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(613).size(), 0);

  // Chamber 7
  // 7R = 7I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(700).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(701).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(702).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(703).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(704).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(705).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(706).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(720).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(721).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(722).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(723).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(724).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(725).size(), 4);
  // 7L = 7O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(707).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(708).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(709).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(710).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(711).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(712).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(713).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(714).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(715).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(716).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(717).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(718).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(719).size(), 1);

  // Chamber 8
  // 8R = 8I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(800).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(801).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(802).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(803).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(804).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(805).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(806).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(820).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(821).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(822).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(823).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(824).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(825).size(), 0);
  // 8L = 8O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(807).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(808).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(809).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(810).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(811).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(812).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(813).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(814).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(815).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(816).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(817).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(818).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(819).size(), 0);

  // Chamber 9
  // 9R = 9I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(900).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(901).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(902).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(903).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(904).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(905).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(906).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(920).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(921).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(922).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(923).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(924).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(925).size(), 0);
  // 9L = 9O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(907).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(908).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(909).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(910).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(911).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(912).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(913).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(914).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(915).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(916).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(917).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(918).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(919).size(), 0);

  // Chamber 10
  // 10R = 10I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1000).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1001).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1002).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1003).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1004).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1005).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1006).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1020).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1021).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1022).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1023).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1024).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1025).size(), 0);
  // 10L = 10O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1007).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1008).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1009).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1010).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1011).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1012).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1013).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1014).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1015).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1016).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1017).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1018).size(), 0);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1019).size(), 0);
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AllSolarsMustGetACruLink, T, realTypes)
{
  std::set<uint16_t> solarIds = getSolarUIDs<T>();
  auto solar2cruLink = o2::mch::raw::createSolar2CruLinkMapper<T>();
  int nbad{0};
  for (auto s : solarIds) {
    auto p = solar2cruLink(s);
    if (!p.has_value()) {
      ++nbad;
    }
    // std::cout << fmt::format("SOLAR {:4d} ", s);
    // if (p.has_value()) {
    //   std::cout << fmt::format("CRU {:4d} LINK {:1d}\n", p->cruId(), p->linkId());
    // } else {
    //   std::cout << " NO DATA\n";
    // }
  }
  BOOST_CHECK_EQUAL(nbad, 0);
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
