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

// // Used to generate dslist.cxx
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

  auto d2e = o2::mch::raw::createDet2ElecMapper<T>();

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
      } else {
        std::cout << "did not find matching dsel for id=" << id << "\n";
      }
    };
    if (dsForOneDE.size() != nref) {
      std::cout << fmt::format("ERROR : mapper has {:4d} DS while DE {:4d} should have {:4d}\n", dsForOneDE.size(), deid, nref);
    }
  }
  return ds;
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

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH6L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH6L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH6L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

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

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH8R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH8R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH8R);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH8L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH8L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH8L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH9R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH9R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH9R);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH9L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH9L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH9L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH10R, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH10R);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH10R);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

BOOST_TEST_DECORATOR(*boost::unit_test::disabled())
BOOST_AUTO_TEST_CASE_TEMPLATE(MustContainAllSampaCH10L, T, testTypes)
{
  auto check = nofDualSampasFromMapper<T>(o2::mch::raw::deIdsOfCH10L);
  auto expected = nofDualSampas(o2::mch::raw::deIdsOfCH10L);
  BOOST_CHECK(std::equal(expected.begin(), expected.end(), check.begin()));
}

// this check depends on the installation status at Pt2, i.e.
// not yet fullly installed chambers have an expected number of solarIds
// set to zero
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
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(800).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(801).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(802).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(803).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(804).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(805).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(806).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(820).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(821).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(822).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(823).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(824).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(825).size(), 4);
  // 8L = 8O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(807).size(), 1);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(808).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(809).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(810).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(811).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(812).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(813).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(814).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(815).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(816).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(817).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(818).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(819).size(), 1);

  // Chamber 9
  // 9R = 9I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(900).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(901).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(902).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(903).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(904).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(905).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(906).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(920).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(921).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(922).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(923).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(924).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(925).size(), 4);
  // 9L = 9O
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(907).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(908).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(909).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(910).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(911).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(912).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(913).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(914).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(915).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(916).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(917).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(918).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(919).size(), 2);

  // Chamber 10
  // 10R = 10I
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1000).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1001).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1002).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1003).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1004).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1005).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1006).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1020).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1021).size(), 2);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1022).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1023).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1024).size(), 4);
  BOOST_CHECK_EQUAL(getSolarUIDs<T>(1025).size(), 4);
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

template <typename T>
int expectedNumberOfSolars();

template <>
int expectedNumberOfSolars<ElectronicMapperDummy>()
{
  return 421;
}
template <>
int expectedNumberOfSolars<ElectronicMapperGenerated>()
{
  return 362;
}

template <typename T>
int expectedNumberOfDs();

template <>
int expectedNumberOfDs<ElectronicMapperDummy>()
{
  return 16828;
}

template <>
int expectedNumberOfDs<ElectronicMapperGenerated>()
{
  return 8112;
}

BOOST_AUTO_TEST_CASE_TEMPLATE(AllSolarsMustGetAFeeLinkAndTheReverse, T, testTypes)
{
  auto errors = solar2FeeLinkConsistencyCheck<T>();
  BOOST_CHECK_EQUAL(errors.size(), 0);
  if (!errors.empty()) {
    for (auto msg : errors) {
      std::cout << "ERROR: " << msg << "\n";
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckNumberOfSolars, T, testTypes)
{
  std::set<uint16_t> solarIds = getSolarUIDs<T>();
  BOOST_CHECK_EQUAL(solarIds.size(), expectedNumberOfSolars<T>()); // must be updated when adding more chambers
}

BOOST_AUTO_TEST_CASE_TEMPLATE(CheckNumberOfDsElecId, T, testTypes)
{
  std::set<DsElecId> dsElecIds = getAllDs<T>();
  BOOST_CHECK_EQUAL(dsElecIds.size(), expectedNumberOfDs<T>()); // must be updated when adding more chambers
}

// Spot check (on a few selected ones, e.g. the ones used in some unit tests)
// solars actually have an associated FeeLinkId.
BOOST_AUTO_TEST_CASE_TEMPLATE(CheckAFewSolarIdThatMustHaveAFeeLinkd, T, testTypes)
{
  auto solarIds = {361, 448, 728};
  for (auto solarId : solarIds) {
    BOOST_TEST_INFO(fmt::format("solarId={}", solarId));
    auto s2f = o2::mch::raw::createSolar2FeeLinkMapper<T>();
    auto f = s2f(solarId);
    BOOST_CHECK_EQUAL(f.has_value(), true);
  }
}

BOOST_AUTO_TEST_CASE(SpotCheck)
{
  o2::mch::raw::FeeLinkId id(44, 0);
  auto f2s = o2::mch::raw::createFeeLink2SolarMapper<ElectronicMapperGenerated>();
  auto s = f2s(id);
  BOOST_CHECK_EQUAL(s.has_value(), true);
}
