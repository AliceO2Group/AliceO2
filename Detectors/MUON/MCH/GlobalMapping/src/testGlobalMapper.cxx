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

#include "MCHMappingInterface/CathodeSegmentation.h"
#define BOOST_TEST_MODULE Test MCHGlobalMapping Mapper
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include "MCHConditions/DCSAliases.h"
#include "MCHGlobalMapping/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"

using namespace o2::mch::dcs;

namespace
{

/* Build the list of expected dual sampa indices from a set of Cathodes */
std::set<int> getExpectedDualSampas(const std::set<Cathode>& cathodes)
{
  std::set<int> expectedDualSampas;
  for (const auto& expectedCathode : cathodes) {
    auto deId = expectedCathode.deId;
    bool bending = expectedCathode.plane == Plane::Bending;
    bool checkPlane = expectedCathode.plane != Plane::Both;
    if (checkPlane) {
      o2::mch::mapping::CathodeSegmentation cathode(deId, bending);
      for (auto i = 0; i < cathode.nofDualSampas(); i++) {
        int index = o2::mch::getDsIndex({deId, cathode.dualSampaId(i)});
        expectedDualSampas.emplace(index);
      }
    } else {
      const o2::mch::mapping::Segmentation& seg = o2::mch::mapping::segmentation(deId);
      seg.forEachDualSampa([&expectedDualSampas, deId](int dualSampaId) {
        int index = o2::mch::getDsIndex({deId, dualSampaId});
        expectedDualSampas.emplace(index);
      });
    }
  }
  return expectedDualSampas;
}

/* Build the list of expected dual sampa indices from a set of solar Ids */
std::set<int> getExpectedDualSampas(const std::set<int>& solarIds)
{
  std::set<int> expectedDualSampas;
  for (const auto& solarId : solarIds) {
    auto dsDetIds = o2::mch::raw::getDualSampas<o2::mch::raw::ElectronicMapperGenerated>(solarId);
    for (const auto& dsDetId : dsDetIds) {
      int index = o2::mch::getDsIndex(dsDetId);
      expectedDualSampas.emplace(index);
    }
  }
  return expectedDualSampas;
}

std::string expandAlias(std::string_view shortAlias)
{
  std::string alias = shortAlias.find("Left") != std::string::npos ? "MchHvLvLeft" : "MchHvLvRight";
  alias += "/" + std::string(shortAlias);
  if (alias.find("actual") == std::string::npos) {
    alias += ".SenseVoltage";
  }
  return alias;
}

/* Compare the result of the aliasToDsIndices function with expectations */
template <typename T>
void compareToExpectation(const std::map<std::string, std::set<T>>& expected)
{
  for (auto e : expected) {

    // build the full alias from the short form in the expected map
    std::string alias = expandAlias(e.first);

    // this is the function we are testing
    auto dsix = o2::mch::dcs::aliasToDsIndices(alias);

    std::set<int> expectedDualSampas = getExpectedDualSampas(e.second);

    BOOST_TEST_CONTEXT(fmt::format("alias {}", alias))
    {
      // finally compare to expectations
      BOOST_REQUIRE_EQUAL(dsix.size(), expectedDualSampas.size());
      bool permutation = std::is_permutation(begin(dsix), end(dsix), begin(expectedDualSampas));
      BOOST_REQUIRE_EQUAL(permutation, true);
    }
  }
}

} // namespace

BOOST_AUTO_TEST_CASE(HVSlatIDToDsIndex)
{
  std::map<std::string, std::set<Cathode>> expected = {
    {"Chamber09Right/Slat00.actual.vMon", {{1006, Plane::Both}}},
  };

  compareToExpectation(expected);
}

BOOST_AUTO_TEST_CASE(HVSectorIDToDsIndex)
{
  auto indices1 = o2::mch::dcs::aliasToDsIndices("MchHvLvRight/Chamber00Right/Quad0Sect0.actual.vMon");
  auto indices2 = o2::mch::dcs::aliasToDsIndices("MchHvLvRight/Chamber00Right/Quad0Sect1.actual.vMon");
  auto indices3 = o2::mch::dcs::aliasToDsIndices("MchHvLvRight/Chamber00Right/Quad0Sect2.actual.vMon");

  o2::mch::mapping::Segmentation de(100);

  BOOST_CHECK_EQUAL(indices1.size() + indices2.size() + indices3.size(), de.nofDualSampas());
  BOOST_CHECK_EQUAL(indices1.size(), 205);
  BOOST_CHECK_EQUAL(indices2.size(), 184);
  BOOST_CHECK_EQUAL(indices3.size(), 62);
}

BOOST_AUTO_TEST_CASE(LVAliasToSolar)
{
  /* For St345, solar crates might not correspond to full detection elements
   * cathode like for St12. Hence we have to test explicitely the list of
   * solarIds handled by a given alias.
   *
   * Note that the few cases where a full cathode is handled by a St345
   * solar alias is tested both in this test and in the next one
   * (LVAliasToCathode), as stated in the comments below in the "expected" map.
   *
   * Note that the solarIds list is explicitely expanded, even if could
   * have been represented by e.g. {first value, length} as the numbers
   * are always contiguous), to allow grepping (the solarIds are unique ids)
   * if need be (in the "test code is documentation" philosophy)
   */
  std::map<std::string, std::set<int>> expected = {

    /* Left */

    {"Chamber04Left/SolCh04LCr01", {144, 145, 146, 147, 148}},
    {"Chamber04Left/SolCh04LCr02", {80, 81, 82, 83, 84, 85}},
    {"Chamber04Left/SolCh04LCr03", {424, 425, 426, 427, 428, 429}},
    {"Chamber04Left/SolCh04LCr04", {440, 441, 442, 443, 444, 445}},
    {"Chamber04Left/SolCh04LCr05", {336, 337, 338}}, /* see also (LVAliasToCathode) */

    {"Chamber05Left/SolCh05LCr01", {312, 313, 314}}, /* see also (LVAliasToCathode) */
    {"Chamber05Left/SolCh05LCr02", {8, 9, 10, 11, 12, 13}},
    {"Chamber05Left/SolCh05LCr03", {56, 57, 58, 59, 60, 61}},
    {"Chamber05Left/SolCh05LCr04", {32, 33, 34, 35, 36, 37}},
    {"Chamber05Left/SolCh05LCr05", {24, 25, 26, 27, 28}},

    {"Chamber06Left/SolCh06LCr01", {720, 721, 722, 723, 724}}, /* see also (LVAliasToCathode) */
    {"Chamber06Left/SolCh06LCr02", {920, 921, 922, 923, 924, 925}},
    {"Chamber06Left/SolCh06LCr03", {784, 785, 786, 787, 788, 789}},
    {"Chamber06Left/SolCh06LCr04", {912, 913, 914, 915}}, /* see also (LVAliasToCathode) */
    {"Chamber06Left/SolCh06LCr05", {328, 329, 330, 331, 332, 333}},
    {"Chamber06Left/SolCh06LCr06", {344, 345, 346, 347, 348, 349}},
    {"Chamber06Left/SolCh06LCr07", {848, 849, 850, 851, 852}}, /* see also (LVAliasToCathode) */

    {"Chamber07Left/SolCh07LCr01", {728, 729, 730, 731, 732}}, /* see also (LVAliasToCathode) */
    {"Chamber07Left/SolCh07LCr02", {736, 737, 738, 739, 740, 741}},
    {"Chamber07Left/SolCh07LCr03", {776, 777, 778, 779, 780, 781}},
    {"Chamber07Left/SolCh07LCr04", {864, 865, 866, 867}}, /* see also (LVAliasToCathode) */
    {"Chamber07Left/SolCh07LCr05", {352, 353, 354, 355, 356, 357}},
    {"Chamber07Left/SolCh07LCr06", {304, 305, 306, 307, 308, 309}},
    {"Chamber07Left/SolCh07LCr07", {856, 857, 858, 859, 860}}, /* see also (LVAliasToCathode) */

    {"Chamber08Left/SolCh08LCr01", {680, 681, 682, 683}}, /* see also (LVAliasToCathode) */
    {"Chamber08Left/SolCh08LCr02", {744, 745, 746, 747, 748, 749}},
    {"Chamber08Left/SolCh08LCr03", {752, 753, 754, 755, 756, 757}},
    {"Chamber08Left/SolCh08LCr04", {696, 697, 698, 699, 700, 701}},
    {"Chamber08Left/SolCh08LCr05", {704, 705, 706, 707, 708, 709}},
    {"Chamber08Left/SolCh08LCr06", {632, 633, 634, 635, 636, 637}},
    {"Chamber08Left/SolCh08LCr07", {880, 881, 882, 883, 884, 885}},
    {"Chamber08Left/SolCh08LCr08", {872, 873, 874, 875}}, /* see also (LVAliasToCathode) */

    {"Chamber09Left/SolCh09LCr01", {640, 641, 642, 643}}, /* see also (LVAliasToCathode) */
    {"Chamber09Left/SolCh09LCr02", {712, 713, 714, 715, 716, 717}},
    {"Chamber09Left/SolCh09LCr03", {768, 769, 770, 771, 772, 773}},
    {"Chamber09Left/SolCh09LCr04", {656, 657, 658, 659, 660, 661}},
    {"Chamber09Left/SolCh09LCr05", {760, 761, 762, 763, 764, 765}},
    {"Chamber09Left/SolCh09LCr06", {904, 905, 906, 907, 908, 909}},
    {"Chamber09Left/SolCh09LCr07", {888, 889, 890, 891, 892, 893}},
    {"Chamber09Left/SolCh09LCr08", {896, 897, 898, 899}}, /* see also (LVAliasToCathode) */

    /* Right */

    {"Chamber04Right/SolCh04RCr01", {72, 73, 74, 75, 76}},
    {"Chamber04Right/SolCh04RCr02", {184, 185, 186, 187, 188, 189}},
    {"Chamber04Right/SolCh04RCr03", {456, 457, 458, 459, 460, 461}},
    {"Chamber04Right/SolCh04RCr04", {400, 401, 402, 403, 404, 405}},
    {"Chamber04Right/SolCh04RCr05", {368, 369, 370}}, /* see also (LVAliasToCathode) */

    {"Chamber05Right/SolCh05RCr01", {448, 449, 450}}, /* see also (LVAliasToCathode) */
    {"Chamber05Right/SolCh05RCr02", {360, 361, 362, 363, 364, 365}},
    {"Chamber05Right/SolCh05RCr03", {216, 217, 218, 219, 220, 221}},
    {"Chamber05Right/SolCh05RCr04", {432, 433, 434, 435, 436, 437}},
    {"Chamber05Right/SolCh05RCr05", {408, 409, 410, 411, 412}},

    {"Chamber06Right/SolCh06RCr01", {840, 841, 842, 843, 844}}, /* see also (LVAliasToCathode) */
    {"Chamber06Right/SolCh06RCr02", {800, 801, 802, 803, 804, 805}},
    {"Chamber06Right/SolCh06RCr03", {816, 817, 818, 819, 820, 821}},
    {"Chamber06Right/SolCh06RCr04", {624, 625, 626, 627}}, /* see also (LVAliasToCathode) */
    {"Chamber06Right/SolCh06RCr05", {528, 529, 530, 531, 532, 533}},
    {"Chamber06Right/SolCh06RCr06", {512, 513, 514, 515, 516, 517}},
    {"Chamber06Right/SolCh06RCr07", {584, 585, 586, 587, 588}}, /* see also (LVAliasToCathode) */

    {"Chamber07Right/SolCh07RCr01", {824, 825, 826, 827, 828}}, /* see also (LVAliasToCathode) */
    {"Chamber07Right/SolCh07RCr02", {808, 809, 810, 811, 812, 813}},
    {"Chamber07Right/SolCh07RCr03", {832, 833, 834, 835, 836, 837}},
    {"Chamber07Right/SolCh07RCr04", {600, 601, 602, 603}}, /* see also (LVAliasToCathode) */
    {"Chamber07Right/SolCh07RCr05", {576, 577, 578, 579, 580, 581}},
    {"Chamber07Right/SolCh07RCr06", {592, 593, 594, 595, 596, 597}},
    {"Chamber07Right/SolCh07RCr07", {496, 497, 498, 499, 500}}, /* see also (LVAliasToCathode) */

    {"Chamber08Right/SolCh08RCr01", {688, 689, 690, 691}}, /* see also (LVAliasToCathode) */
    {"Chamber08Right/SolCh08RCr02", {672, 673, 674, 675, 676, 677}},
    {"Chamber08Right/SolCh08RCr03", {560, 561, 562, 563, 564, 565}},
    {"Chamber08Right/SolCh08RCr04", {568, 569, 570, 571, 572, 573}},
    {"Chamber08Right/SolCh08RCr05", {608, 609, 610, 611, 612, 613}},
    {"Chamber08Right/SolCh08RCr06", {616, 617, 618, 619, 620, 621}},
    {"Chamber08Right/SolCh08RCr07", {480, 481, 482, 483, 484, 485}},
    {"Chamber08Right/SolCh08RCr08", {488, 489, 490, 491}}, /* see also (LVAliasToCathode) */

    {"Chamber09Right/SolCh09RCr01", {648, 649, 650, 651}}, /* see also (LVAliasToCathode) */
    {"Chamber09Right/SolCh09RCr02", {664, 665, 666, 667, 668, 669}},
    {"Chamber09Right/SolCh09RCr03", {552, 553, 554, 555, 556, 557}},
    {"Chamber09Right/SolCh09RCr04", {504, 505, 506, 507, 508, 509}},
    {"Chamber09Right/SolCh09RCr05", {536, 537, 538, 539, 540, 541}},
    {"Chamber09Right/SolCh09RCr06", {544, 545, 546, 547, 548, 549}},
    {"Chamber09Right/SolCh09RCr07", {520, 521, 522, 523, 524, 525}},
    {"Chamber09Right/SolCh09RCr08", {472, 473, 474, 475}}, /* see also (LVAliasToCathode) */

  };

  compareToExpectation(expected);
}

BOOST_AUTO_TEST_CASE(LVAliasToCathode)
{
  /** the mapping of LV group to quadrant plane and/or slat(s)
   * is not completely intuitive... so we test it as fully as we can here,
   * even if it's a bit manual (but at least we _see_ what is expected).
   *
   * For each alias we test that :
   *
   * - we can the right number of dual sampa indices
   * - those indices matches the right {deId,plane} pair
   *
   * Note the measurement is not relevant in this game,
   * only the chamber/group is.
   *
   * Note also we are only testing the relationships where the destination
   * is either a full detection element or a list of full detection elements.
   *
   * For mappings with partial detection element content, see LVAliasToSolar
   *
   */
  std::map<std::string, std::set<Cathode>> expected = {

    /* Left */

    {"Chamber00Left/Group04an", {{101, Plane::NonBending}}},
    {"Chamber00Left/Group02an", {{101, Plane::Bending}}},
    {"Chamber00Left/Group03an", {{102, Plane::Bending}}},
    {"Chamber00Left/Group01an", {{102, Plane::NonBending}}},
    {"Chamber00Left/SolCh00LCr01", {{101, Plane::NonBending}}},
    {"Chamber00Left/SolCh00LCr02", {{101, Plane::Bending}}},
    {"Chamber00Left/SolCh00LCr03", {{102, Plane::Bending}}},
    {"Chamber00Left/SolCh00LCr04", {{102, Plane::NonBending}}},

    {"Chamber01Left/Group04an", {{201, Plane::NonBending}}},
    {"Chamber01Left/Group02an", {{201, Plane::Bending}}},
    {"Chamber01Left/Group03an", {{202, Plane::Bending}}},
    {"Chamber01Left/Group01an", {{202, Plane::NonBending}}},
    {"Chamber01Left/SolCh01LCr01", {{201, Plane::NonBending}}},
    {"Chamber01Left/SolCh01LCr02", {{201, Plane::Bending}}},
    {"Chamber01Left/SolCh01LCr03", {{202, Plane::Bending}}},
    {"Chamber01Left/SolCh01LCr04", {{202, Plane::NonBending}}},

    {"Chamber02Left/Group02an", {{301, Plane::NonBending}}},
    {"Chamber02Left/Group01an", {{301, Plane::Bending}}},
    {"Chamber02Left/Group04an", {{302, Plane::Bending}}},
    {"Chamber02Left/Group03an", {{302, Plane::NonBending}}},
    {"Chamber02Left/SolCh02LCr01", {{301, Plane::NonBending}}},
    {"Chamber02Left/SolCh02LCr02", {{301, Plane::Bending}}},
    {"Chamber02Left/SolCh02LCr03", {{302, Plane::Bending}}},
    {"Chamber02Left/SolCh02LCr04", {{302, Plane::NonBending}}},

    {"Chamber03Left/Group02an", {{401, Plane::NonBending}}},
    {"Chamber03Left/Group01an", {{401, Plane::Bending}}},
    {"Chamber03Left/Group04an", {{402, Plane::Bending}}},
    {"Chamber03Left/Group03an", {{402, Plane::NonBending}}},
    {"Chamber03Left/SolCh03LCr01", {{401, Plane::NonBending}}},
    {"Chamber03Left/SolCh03LCr02", {{401, Plane::Bending}}},
    {"Chamber03Left/SolCh03LCr03", {{402, Plane::Bending}}},
    {"Chamber03Left/SolCh03LCr04", {{402, Plane::NonBending}}},

    {"Chamber04Left/Group05an", {{511, Plane::Both}, {512, Plane::Both}, {513, Plane::Both}}},
    {"Chamber04Left/Group04an", {{510, Plane::Both}}},
    {"Chamber04Left/Group03an", {{509, Plane::Both}}},
    {"Chamber04Left/Group02an", {{508, Plane::Both}}},
    {"Chamber04Left/Group01an", {{505, Plane::Both}, {506, Plane::Both}, {507, Plane::Both}}},
    {"Chamber04Left/SolCh04LCr05", {{512, Plane::Both}, {513, Plane::Both}}},

    {"Chamber05Left/Group05an", {{611, Plane::Both}, {612, Plane::Both}, {613, Plane::Both}}},
    {"Chamber05Left/Group04an", {{610, Plane::Both}}},
    {"Chamber05Left/Group03an", {{609, Plane::Both}}},
    {"Chamber05Left/Group02an", {{608, Plane::Both}}},
    {"Chamber05Left/Group01an", {{605, Plane::Both}, {606, Plane::Both}, {607, Plane::Both}}},
    {"Chamber05Left/SolCh05LCr01", {{605, Plane::Both}, {606, Plane::Both}}},

    {"Chamber06Left/Group07an", {{716, Plane::Both}, {717, Plane::Both}, {718, Plane::Both}, {719, Plane::Both}}},
    {"Chamber06Left/Group06an", {{715, Plane::Both}}},
    {"Chamber06Left/Group05an", {{714, Plane::Both}}},
    {"Chamber06Left/Group04an", {{713, Plane::Both}}},
    {"Chamber06Left/Group03an", {{712, Plane::Both}}},
    {"Chamber06Left/Group02an", {{711, Plane::Both}}},
    {"Chamber06Left/Group01an", {{707, Plane::Both}, {708, Plane::Both}, {709, Plane::Both}, {710, Plane::Both}}},
    {"Chamber06Left/SolCh06LCr01", {{707, Plane::Both}, {708, Plane::Both}, {709, Plane::Both}}},
    {"Chamber06Left/SolCh06LCr04", {{713, Plane::Both}}},
    {"Chamber06Left/SolCh06LCr07", {{717, Plane::Both}, {718, Plane::Both}, {719, Plane::Both}}},

    {"Chamber07Left/Group07an", {{816, Plane::Both}, {817, Plane::Both}, {818, Plane::Both}, {819, Plane::Both}}},
    {"Chamber07Left/Group06an", {{815, Plane::Both}}},
    {"Chamber07Left/Group05an", {{814, Plane::Both}}},
    {"Chamber07Left/Group04an", {{813, Plane::Both}}},
    {"Chamber07Left/Group03an", {{812, Plane::Both}}},
    {"Chamber07Left/Group02an", {{811, Plane::Both}}},
    {"Chamber07Left/Group01an", {{807, Plane::Both}, {808, Plane::Both}, {809, Plane::Both}, {810, Plane::Both}}},
    {"Chamber07Left/SolCh07LCr01", {{807, Plane::Both}, {808, Plane::Both}, {809, Plane::Both}}},
    {"Chamber07Left/SolCh07LCr04", {{813, Plane::Both}}},
    {"Chamber07Left/SolCh07LCr07", {{817, Plane::Both}, {818, Plane::Both}, {819, Plane::Both}}},

    {"Chamber08Left/Group07an", {{916, Plane::Both}, {917, Plane::Both}, {918, Plane::Both}, {919, Plane::Both}}},
    {"Chamber08Left/Group06an", {{915, Plane::Both}}},
    {"Chamber08Left/Group05an", {{914, Plane::Both}}},
    {"Chamber08Left/Group04an", {{913, Plane::Both}}},
    {"Chamber08Left/Group03an", {{912, Plane::Both}}},
    {"Chamber08Left/Group02an", {{911, Plane::Both}}},
    {"Chamber08Left/Group01an", {{907, Plane::Both}, {908, Plane::Both}, {909, Plane::Both}, {910, Plane::Both}}},
    {"Chamber08Left/SolCh08LCr01", {{907, Plane::Both}, {908, Plane::Both}}},
    {"Chamber08Left/SolCh08LCr08", {{918, Plane::Both}, {919, Plane::Both}}},

    {"Chamber09Left/Group07an", {{1016, Plane::Both}, {1017, Plane::Both}, {1018, Plane::Both}, {1019, Plane::Both}}},
    {"Chamber09Left/Group06an", {{1015, Plane::Both}}},
    {"Chamber09Left/Group05an", {{1014, Plane::Both}}},
    {"Chamber09Left/Group04an", {{1013, Plane::Both}}},
    {"Chamber09Left/Group03an", {{1012, Plane::Both}}},
    {"Chamber09Left/Group02an", {{1011, Plane::Both}}},
    {"Chamber09Left/Group01an", {{1007, Plane::Both}, {1008, Plane::Both}, {1009, Plane::Both}, {1010, Plane::Both}}},
    {"Chamber09Left/SolCh09LCr01", {{1007, Plane::Both}, {1008, Plane::Both}}},
    {"Chamber09Left/SolCh09LCr08", {{1018, Plane::Both}, {1019, Plane::Both}}},

    /* Right */

    {"Chamber00Right/Group04an", {{100, Plane::Bending}}},
    {"Chamber00Right/Group02an", {{100, Plane::NonBending}}},
    {"Chamber00Right/Group03an", {{103, Plane::NonBending}}},
    {"Chamber00Right/Group01an", {{103, Plane::Bending}}},
    {"Chamber00Right/SolCh00RCr01", {{100, Plane::Bending}}},
    {"Chamber00Right/SolCh00RCr02", {{100, Plane::NonBending}}},
    {"Chamber00Right/SolCh00RCr03", {{103, Plane::NonBending}}},
    {"Chamber00Right/SolCh00RCr04", {{103, Plane::Bending}}},

    {"Chamber01Right/Group04an", {{200, Plane::Bending}}},
    {"Chamber01Right/Group02an", {{200, Plane::NonBending}}},
    {"Chamber01Right/Group03an", {{203, Plane::NonBending}}},
    {"Chamber01Right/Group01an", {{203, Plane::Bending}}},
    {"Chamber01Right/SolCh01RCr01", {{200, Plane::Bending}}},
    {"Chamber01Right/SolCh01RCr02", {{200, Plane::NonBending}}},
    {"Chamber01Right/SolCh01RCr03", {{203, Plane::NonBending}}},
    {"Chamber01Right/SolCh01RCr04", {{203, Plane::Bending}}},

    {"Chamber02Right/Group02an", {{300, Plane::Bending}}},
    {"Chamber02Right/Group01an", {{300, Plane::NonBending}}},
    {"Chamber02Right/Group04an", {{303, Plane::NonBending}}},
    {"Chamber02Right/Group03an", {{303, Plane::Bending}}},
    {"Chamber02Right/SolCh02RCr01", {{300, Plane::Bending}}},
    {"Chamber02Right/SolCh02RCr02", {{300, Plane::NonBending}}},
    {"Chamber02Right/SolCh02RCr03", {{303, Plane::NonBending}}},
    {"Chamber02Right/SolCh02RCr04", {{303, Plane::Bending}}},

    {"Chamber03Right/Group02an", {{400, Plane::Bending}}},
    {"Chamber03Right/Group01an", {{400, Plane::NonBending}}},
    {"Chamber03Right/Group04an", {{403, Plane::NonBending}}},
    {"Chamber03Right/Group03an", {{403, Plane::Bending}}},
    {"Chamber03Right/SolCh03RCr01", {{400, Plane::Bending}}},
    {"Chamber03Right/SolCh03RCr02", {{400, Plane::NonBending}}},
    {"Chamber03Right/SolCh03RCr03", {{403, Plane::NonBending}}},
    {"Chamber03Right/SolCh03RCr04", {{403, Plane::Bending}}},

    {"Chamber04Right/Group05an", {{514, Plane::Both}, {515, Plane::Both}, {516, Plane::Both}}},
    {"Chamber04Right/Group04an", {{517, Plane::Both}}},
    {"Chamber04Right/Group03an", {{500, Plane::Both}}},
    {"Chamber04Right/Group02an", {{501, Plane::Both}}},
    {"Chamber04Right/Group01an", {{502, Plane::Both}, {503, Plane::Both}, {504, Plane::Both}}},
    {"Chamber04Right/SolCh04RCr05", {{514, Plane::Both}, {515, Plane::Both}}},

    {"Chamber05Right/Group05an", {{614, Plane::Both}, {615, Plane::Both}, {616, Plane::Both}}},
    {"Chamber05Right/Group04an", {{617, Plane::Both}}},
    {"Chamber05Right/Group03an", {{600, Plane::Both}}},
    {"Chamber05Right/Group02an", {{601, Plane::Both}}},
    {"Chamber05Right/Group01an", {{602, Plane::Both}, {603, Plane::Both}, {604, Plane::Both}}},
    {"Chamber05Right/SolCh05RCr01", {{603, Plane::Both}, {604, Plane::Both}}},

    {"Chamber06Right/Group07an", {{720, Plane::Both}, {721, Plane::Both}, {722, Plane::Both}, {723, Plane::Both}}},
    {"Chamber06Right/Group06an", {{724, Plane::Both}}},
    {"Chamber06Right/Group05an", {{725, Plane::Both}}},
    {"Chamber06Right/Group04an", {{700, Plane::Both}}},
    {"Chamber06Right/Group03an", {{701, Plane::Both}}},
    {"Chamber06Right/Group02an", {{702, Plane::Both}}},
    {"Chamber06Right/Group01an", {{703, Plane::Both}, {704, Plane::Both}, {705, Plane::Both}, {706, Plane::Both}}},
    {"Chamber06Right/SolCh06RCr01", {{704, Plane::Both}, {705, Plane::Both}, {706, Plane::Both}}},
    {"Chamber06Right/SolCh06RCr04", {{700, Plane::Both}}},
    {"Chamber06Right/SolCh06RCr07", {{720, Plane::Both}, {721, Plane::Both}, {722, Plane::Both}}},

    {"Chamber07Right/Group07an", {{820, Plane::Both}, {821, Plane::Both}, {822, Plane::Both}, {823, Plane::Both}}},
    {"Chamber07Right/Group06an", {{824, Plane::Both}}},
    {"Chamber07Right/Group05an", {{825, Plane::Both}}},
    {"Chamber07Right/Group04an", {{800, Plane::Both}}},
    {"Chamber07Right/Group03an", {{801, Plane::Both}}},
    {"Chamber07Right/Group02an", {{802, Plane::Both}}},
    {"Chamber07Right/Group01an", {{803, Plane::Both}, {804, Plane::Both}, {805, Plane::Both}, {806, Plane::Both}}},
    {"Chamber07Right/SolCh07RCr01", {{804, Plane::Both}, {805, Plane::Both}, {806, Plane::Both}}},
    {"Chamber07Right/SolCh07RCr04", {{800, Plane::Both}}},
    {"Chamber07Right/SolCh07RCr07", {{820, Plane::Both}, {821, Plane::Both}, {822, Plane::Both}}},

    {"Chamber08Right/Group07an", {{920, Plane::Both}, {921, Plane::Both}, {922, Plane::Both}, {923, Plane::Both}}},
    {"Chamber08Right/Group06an", {{924, Plane::Both}}},
    {"Chamber08Right/Group05an", {{925, Plane::Both}}},
    {"Chamber08Right/Group04an", {{900, Plane::Both}}},
    {"Chamber08Right/Group03an", {{901, Plane::Both}}},
    {"Chamber08Right/Group02an", {{902, Plane::Both}}},
    {"Chamber08Right/Group01an", {{903, Plane::Both}, {904, Plane::Both}, {905, Plane::Both}, {906, Plane::Both}}},
    {"Chamber08Right/SolCh08RCr01", {{905, Plane::Both}, {906, Plane::Both}}},
    {"Chamber08Right/SolCh08RCr08", {{920, Plane::Both}, {921, Plane::Both}}},

    {"Chamber09Right/Group07an", {{1020, Plane::Both}, {1021, Plane::Both}, {1022, Plane::Both}, {1023, Plane::Both}}},
    {"Chamber09Right/Group06an", {{1024, Plane::Both}}},
    {"Chamber09Right/Group05an", {{1025, Plane::Both}}},
    {"Chamber09Right/Group04an", {{1000, Plane::Both}}},
    {"Chamber09Right/Group03an", {{1001, Plane::Both}}},
    {"Chamber09Right/Group02an", {{1002, Plane::Both}}},
    {"Chamber09Right/Group01an", {{1003, Plane::Both}, {1004, Plane::Both}, {1005, Plane::Both}, {1006, Plane::Both}}},
    {"Chamber09Right/SolCh09RCr01", {{1005, Plane::Both}, {1006, Plane::Both}}},
    {"Chamber09Right/SolCh09RCr08", {{1020, Plane::Both}, {1021, Plane::Both}}},
  };

  compareToExpectation(expected);
}
