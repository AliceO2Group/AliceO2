// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#define BOOST_TEST_MODULE Test MCHSimulation GeoDigiRes
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MAIN

#include <boost/test/unit_test.hpp>

#include "MCHMappingInterface/CathodeSegmentation.h"
#include "MCHSimulation/Geometry.h"
#include "MCHSimulation/GeometryTest.h"
#include "TGeoManager.h"
#include "boost/format.hpp"
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iomanip>
#include <iostream>

namespace bdata = boost::unit_test::data;

struct GEOMETRY {
  GEOMETRY()
  {
    if (!gGeoManager) {
      o2::mch::test::createStandaloneGeometry();
    }
  }
};

const std::array<std::string, 8> quadrantChamberNames{"SC01I", "SC01O", "SC02I", "SC02O", "SC03I", "SC03O",
                                                      "SC04I", "SC04O"};

const std::array<std::string, 12> slatChamberNames{"SC05I", "SC05O", "SC06I", "SC06O", "SC07I", "SC07O",
                                                   "SC08I", "SC08O", "SC09I", "SC09O", "SC10I", "SC10O"};

BOOST_AUTO_TEST_SUITE(o2_mch_simulation)

BOOST_FIXTURE_TEST_SUITE(geometrytransformer, GEOMETRY)

BOOST_AUTO_TEST_CASE(CanGetAllChambers)
{
  std::vector<std::string> chamberNames{quadrantChamberNames.begin(), quadrantChamberNames.end()};

  chamberNames.insert(chamberNames.end(), slatChamberNames.begin(), slatChamberNames.end());

  for (auto chname : chamberNames) {
    auto vol = gGeoManager->GetVolume(chname.c_str());
    BOOST_TEST_REQUIRE((vol != nullptr));
  }
}

std::vector<TGeoNode*> getSlatNodes()
{
  std::vector<TGeoNode*> slats;
  for (auto chname : slatChamberNames) {
    auto vol = gGeoManager->GetVolume(chname.c_str());
    TIter next(vol->GetNodes());
    while (TGeoNode* node = static_cast<TGeoNode*>(next())) {
      if (strstr(node->GetName(), "support") == nullptr) {
        slats.push_back(node);
      }
    }
  }
  return slats;
}

std::vector<TGeoNode*> getQuadrantNodes()
{
  std::vector<TGeoNode*> quadrants;
  for (auto chname : quadrantChamberNames) {
    auto vol = gGeoManager->GetVolume(chname.c_str());
    TIter next(vol->GetNodes());
    while (TGeoNode* node = static_cast<TGeoNode*>(next())) {
      quadrants.push_back(node);
    }
  }
  return quadrants;
}

BOOST_AUTO_TEST_CASE(GetRightNumberOfSlats)
{
  auto slats = getSlatNodes();
  BOOST_CHECK_EQUAL(slats.size(), 140);
}

BOOST_AUTO_TEST_CASE(GetRightNumberOfQuadrants)
{
  auto quadrants = getQuadrantNodes();
  BOOST_CHECK_EQUAL(quadrants.size(), 16);
}

BOOST_AUTO_TEST_CASE(GetDetElemVolumePath, *boost::unit_test::disabled() * boost::unit_test::label("debug"))
{
  TIter next(gGeoManager->GetTopNode()->GetNodes());
  TGeoNode* node;
  TGeoNode* n2;

  std::vector<std::string> codeLines;

  while ((node = static_cast<TGeoNode*>(next()))) {
    std::cout << node->GetName() << "\n";
    TIter next2(node->GetNodes());
    while ((n2 = static_cast<TGeoNode*>(next2()))) {
      std::string n2name{n2->GetName()};
      auto index = n2name.find_last_of('_');
      int detElemId = std::atoi(n2name.substr(index + 1).c_str());
      if (detElemId >= 100) {
        std::stringstream s;
        s << "if (detElemId==" << detElemId << ") {\n";
        s << R"(  return ")" << node->GetName() << "/" << n2name << "\";\n";
        s << "}\n";
        codeLines.push_back(s.str());
      }
    }
  }

  for (auto s : codeLines) {
    std::cout << s;
  }
  BOOST_CHECK_EQUAL(codeLines.size(), 156);
}

BOOST_AUTO_TEST_CASE(GetTransformations)
{
  BOOST_REQUIRE(gGeoManager != nullptr);

  o2::mch::mapping::forEachDetectionElement([](int detElemId) {
    BOOST_CHECK_NO_THROW((o2::mch::getTransformation(detElemId, *gGeoManager)));
  });
}

BOOST_AUTO_TEST_CASE(TextualTreeDump)
{
  const std::string expected =
    R"(cave_1
├──SC01I_0
│  ├──Quadrant (chamber 1)_100
│  └──Quadrant (chamber 1)_103
├──SC01O_1
│  ├──Quadrant (chamber 1)_101
│  └──Quadrant (chamber 1)_102
├──SC02I_2
│  ├──Quadrant (chamber 2)_200
│  └──Quadrant (chamber 2)_203
├──SC02O_3
│  ├──Quadrant (chamber 2)_201
│  └──Quadrant (chamber 2)_202
├──SC03I_4
│  ├──Station 2 quadrant_300
│  └──Station 2 quadrant_303
├──SC03O_5
│  ├──Station 2 quadrant_301
│  └──Station 2 quadrant_302
├──SC04I_6
│  ├──Station 2 quadrant_400
│  └──Station 2 quadrant_403
├──SC04O_7
│  ├──Station 2 quadrant_401
│  └──Station 2 quadrant_402
├──SC05I_8
│  ├──Chamber 5 support panel_8
│  ├──122000SR1_500
│  ├──112200SR2_501
│  ├──122200S_502
│  ├──222000N_503
│  ├──220000N_504
│  ├──220000N_514
│  ├──222000N_515
│  ├──122200S_516
│  └──112200SR2_517
├──SC05O_9
│  ├──Chamber 5 support panel_9
│  ├──220000N_505
│  ├──222000N_506
│  ├──122200S_507
│  ├──112200SR2_508
│  ├──122000SR1_509
│  ├──112200SR2_510
│  ├──122200S_511
│  ├──222000N_512
│  └──220000N_513
├──SC06I_10
│  ├──Chamber 6 support panel_10
│  ├──122000NR1_600
│  ├──112200NR2_601
│  ├──122200N_602
│  ├──222000N_603
│  ├──220000N_604
│  ├──220000N_614
│  ├──222000N_615
│  ├──122200N_616
│  └──112200NR2_617
├──SC06O_11
│  ├──Chamber 6 support panel_11
│  ├──220000N_605
│  ├──222000N_606
│  ├──122200N_607
│  ├──112200NR2_608
│  ├──122000NR1_609
│  ├──112200NR2_610
│  ├──122200N_611
│  ├──222000N_612
│  └──220000N_613
├──SC07I_12
│  ├──Chamber 7 support panel_12
│  ├──122330N_700
│  ├──112233NR3_701
│  ├──112230N_702
│  ├──222330N_703
│  ├──223300N_704
│  ├──333000N_705
│  ├──330000N_706
│  ├──330000N_720
│  ├──333000N_721
│  ├──223300N_722
│  ├──222330N_723
│  ├──112230N_724
│  └──112233NR3_725
├──SC07O_13
│  ├──Chamber 7 support panel_13
│  ├──330000N_707
│  ├──333000N_708
│  ├──223300N_709
│  ├──222330N_710
│  ├──112230N_711
│  ├──112233NR3_712
│  ├──122330N_713
│  ├──112233NR3_714
│  ├──112230N_715
│  ├──222330N_716
│  ├──223300N_717
│  ├──333000N_718
│  └──330000N_719
├──SC08I_14
│  ├──Chamber 8 support panel_14
│  ├──122330N_800
│  ├──112233NR3_801
│  ├──112230N_802
│  ├──222330N_803
│  ├──223300N_804
│  ├──333000N_805
│  ├──330000N_806
│  ├──330000N_820
│  ├──333000N_821
│  ├──223300N_822
│  ├──222330N_823
│  ├──112230N_824
│  └──112233NR3_825
├──SC08O_15
│  ├──Chamber 8 support panel_15
│  ├──330000N_807
│  ├──333000N_808
│  ├──223300N_809
│  ├──222330N_810
│  ├──112230N_811
│  ├──112233NR3_812
│  ├──122330N_813
│  ├──112233NR3_814
│  ├──112230N_815
│  ├──222330N_816
│  ├──223300N_817
│  ├──333000N_818
│  └──330000N_819
├──SC09I_16
│  ├──Chamber 9 support panel_16
│  ├──122330N_900
│  ├──112233NR3_901
│  ├──112233N_902
│  ├──222333N_903
│  ├──223330N_904
│  ├──333300N_905
│  ├──333000N_906
│  ├──333000N_920
│  ├──333300N_921
│  ├──223330N_922
│  ├──222333N_923
│  ├──112233N_924
│  └──112233NR3_925
├──SC09O_17
│  ├──Chamber 9 support panel_17
│  ├──333000N_907
│  ├──333300N_908
│  ├──223330N_909
│  ├──222333N_910
│  ├──112233N_911
│  ├──112233NR3_912
│  ├──122330N_913
│  ├──112233NR3_914
│  ├──112233N_915
│  ├──222333N_916
│  ├──223330N_917
│  ├──333300N_918
│  └──333000N_919
├──SC10I_18
│  ├──Chamber 10 support panel_18
│  ├──122330N_1000
│  ├──112233NR3_1001
│  ├──112233N_1002
│  ├──222333N_1003
│  ├──223330N_1004
│  ├──333300N_1005
│  ├──333000N_1006
│  ├──333000N_1020
│  ├──333300N_1021
│  ├──223330N_1022
│  ├──222333N_1023
│  ├──112233N_1024
│  └──112233NR3_1025
└──SC10O_19
   ├──Chamber 10 support panel_19
   ├──333000N_1007
   ├──333300N_1008
   ├──223330N_1009
   ├──222333N_1010
   ├──112233N_1011
   ├──112233NR3_1012
   ├──122330N_1013
   ├──112233NR3_1014
   ├──112233N_1015
   ├──222333N_1016
   ├──223330N_1017
   ├──333300N_1018
   └──333000N_1019
)";

  std::ostringstream str;
  o2::mch::test::showGeometryAsTextTree("/cave_1", 2, str);
  BOOST_CHECK(expected == str.str());
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
