// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test EMCAL Calib
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "EMCALCalib/BadChannelMap.h"
#include "EMCALBase/Geometry.h"

#include <algorithm>

namespace o2
{

namespace emcal
{

std::vector<unsigned short> getBadChannelsLHC18m_block1();
std::vector<unsigned short> getWarmChannelsLHC18m_block1();
std::vector<unsigned short> getDeadChannelsLHC18m_block1();

std::vector<unsigned short> getBadChannelsLHC17o_block2();
std::vector<unsigned short> getWarmChannelsLHC17o_block2();
std::vector<unsigned short> getDeadChannelsLHC17o_block2();

std::vector<unsigned short> combineMaps(const std::vector<unsigned short>& list1, const std::vector<unsigned short> list2);
void filterBad(std::vector<unsigned short> warmcells, const std::vector<unsigned short> badcells);

BOOST_AUTO_TEST_CASE(BadChannelMap_test)
{
  auto geo = o2::emcal::Geometry::GetInstanceFromRunNumber(300000);

  // Closure creating function testing whether a cell is good, bad
  // or warm. The outer function takes two vectors, (with bad and warm cells),
  // the inner function (used in comparisons) a cell ID.
  auto refChannelCombined = [](const std::vector<unsigned short>& deadcells, const std::vector<unsigned short>& badcells, const std::vector<unsigned short>& warmcells) {
    return [&deadcells, &badcells, &warmcells](unsigned short cellID) {
      if (std::find(deadcells.begin(), deadcells.end(), cellID) != deadcells.end())
        return BadChannelMap::MaskType_t::DEAD_CELL;
      if (std::find(badcells.begin(), badcells.end(), cellID) != badcells.end())
        return BadChannelMap::MaskType_t::BAD_CELL;
      if (std::find(warmcells.begin(), warmcells.end(), cellID) != warmcells.end())
        return BadChannelMap::MaskType_t::WARM_CELL;
      return BadChannelMap::MaskType_t::GOOD_CELL;
    };
  };

  // Single channel test
  //
  // For each channel test set and update, and compare read value
  // against set value
  for (unsigned short c = 0; c < static_cast<unsigned short>(geo->GetNCells()); c++) {
    BadChannelMap singletest;
    singletest.addBadChannel(c, BadChannelMap::MaskType_t::DEAD_CELL);
    BOOST_CHECK_EQUAL(singletest.getChannelStatus(c), BadChannelMap::MaskType_t::DEAD_CELL);
    // update: BAD_CELL
    singletest.addBadChannel(c, BadChannelMap::MaskType_t::BAD_CELL);
    BOOST_CHECK_EQUAL(singletest.getChannelStatus(c), BadChannelMap::MaskType_t::BAD_CELL);
    // update: WARN_CELL
    singletest.addBadChannel(c, BadChannelMap::MaskType_t::WARM_CELL);
    BOOST_CHECK_EQUAL(singletest.getChannelStatus(c), BadChannelMap::MaskType_t::WARM_CELL);
    // update: GOOD_CELL (=erase)
    singletest.addBadChannel(c, BadChannelMap::MaskType_t::GOOD_CELL);
    BOOST_CHECK_EQUAL(singletest.getChannelStatus(c), BadChannelMap::MaskType_t::GOOD_CELL);
  }

  // Pattern test
  //
  // Insert user definde pattern of bad and warm channels, and check for each channel
  // whether the channel is masked or not. The original pattern has to be reconstructed.
  // Test also handles checking good cells.
  //
  // Test data obtained from LHC18m block1
  BadChannelMap patterntest;
  std::vector<unsigned short> badcells = getBadChannelsLHC18m_block1(),
                              deadcells = getDeadChannelsLHC18m_block1(),
                              warmcells = getWarmChannelsLHC18m_block1();
  for (auto bc : badcells) {
    patterntest.addBadChannel(bc, BadChannelMap::MaskType_t::BAD_CELL);
  }
  for (auto wc : warmcells) {
    patterntest.addBadChannel(wc, BadChannelMap::MaskType_t::WARM_CELL);
  }
  for (auto dc : deadcells) {
    patterntest.addBadChannel(dc, BadChannelMap::MaskType_t::DEAD_CELL);
  }
  auto getRefChannelTypeLHC18m = refChannelCombined(deadcells, badcells, warmcells);
  for (unsigned short c = 0; c < static_cast<unsigned short>(geo->GetNCells()); c++) {
    BOOST_CHECK_EQUAL(patterntest.getChannelStatus(c), getRefChannelTypeLHC18m(c));
  }

  // Combine test
  // Second bad channel map is taken from LHC17o block 2
  BadChannelMap lhc17omap;
  std::vector<unsigned short> badLHC17o = getBadChannelsLHC17o_block2(),
                              warmLHC7o = getWarmChannelsLHC17o_block2(),
                              deadLHC17o = getDeadChannelsLHC17o_block2(),
                              combinedBad = combineMaps(badLHC17o, badcells),
                              combinedDead = combineMaps(deadLHC17o, deadcells),
                              combinedWarm = combineMaps(warmLHC7o, warmcells);
  filterBad(combinedWarm, combinedBad);
  filterBad(combinedWarm, combinedDead);
  filterBad(combinedBad, combinedDead);
  for (auto bc : badLHC17o)
    lhc17omap.addBadChannel(bc, BadChannelMap::MaskType_t::BAD_CELL);
  for (auto wc : warmLHC7o)
    lhc17omap.addBadChannel(wc, BadChannelMap::MaskType_t::WARM_CELL);
  for (auto dc : deadLHC17o)
    lhc17omap.addBadChannel(dc, BadChannelMap::MaskType_t::DEAD_CELL);
  BadChannelMap summed(lhc17omap);
  summed += patterntest;
  auto getRefChannelTypeCombined = refChannelCombined(combinedDead, combinedBad, combinedWarm);
  for (unsigned short c = 0; c < static_cast<unsigned short>(geo->GetNCells()); c++) {
    BOOST_CHECK_EQUAL(summed.getChannelStatus(c), getRefChannelTypeCombined(c));
  }

  // Equal
  //
  // - Compare map for LHC17o with itself. The result must be true.
  // - Compare map for LHC18m with map for LHC17o. The result must be false
  BOOST_CHECK_EQUAL(lhc17omap == lhc17omap, true);
  BOOST_CHECK_EQUAL(lhc17omap == patterntest, false);
}

//
// Helper functions
//

std::vector<unsigned short> combineMaps(const std::vector<unsigned short>& list1, const std::vector<unsigned short> list2)
{
  std::vector<unsigned short> combined;
  for (auto l1 : list1)
    combined.emplace_back(l1);
  for (auto l2 : list2) {
    auto found = std::find(combined.begin(), combined.end(), l2);
    if (found == combined.end())
      combined.emplace_back(l2);
  }
  std::sort(combined.begin(), combined.end(), std::less<>());
  return combined;
}

void filterBad(std::vector<unsigned short> warmcells, const std::vector<unsigned short> badcells)
{
  std::vector<unsigned short> todelete;
  for (auto wc : warmcells)
    if (std::find(badcells.begin(), badcells.end(), wc) != badcells.end())
      todelete.emplace_back(wc);
  for (auto td : todelete) {
    auto it = std::find(warmcells.begin(), warmcells.end(), td);
    warmcells.erase(it);
  }
}

//
// Comparison data
//

std::vector<unsigned short> getBadChannelsLHC18m_block1()
{
  return { 103, 128, 152, 176, 191, 198, 287, 324, 328, 353, 384, 386, 387, 397, 399,
           434, 437, 443, 444, 447, 554, 592, 595, 655, 717, 720, 759, 764, 917, 1002,
           1022, 1038, 1050, 1175, 1204, 1222, 1288, 1327, 1329, 1366, 1376, 1380, 1382, 1384, 1386,
           1414, 1519, 1535, 1542, 1693, 1696, 1704, 1711, 1738, 1836, 1837, 1838, 1839, 1844, 1860,
           1867, 1892, 1961, 1963, 2014, 2020, 2022, 2026, 2094, 2126, 2127, 2161, 2193, 2196, 2245,
           2298, 2309, 2313, 2325, 2389, 2395, 2397, 2399, 2406, 2424, 2474, 2487, 2505, 2506, 2533,
           2534, 2540, 2544, 2575, 2581, 2586, 2624, 2665, 2682, 2787, 2797, 2805, 2823, 2824, 2825,
           2841, 2857, 2884, 2888, 2891, 2915, 2921, 2985, 3002, 3039, 3051, 3135, 3161, 3176, 3196,
           3223, 3236, 3244, 3259, 3297, 3307, 3339, 3353, 3397, 3403, 3488, 3493, 3503, 3551, 3732,
           3740, 3748, 3754, 3770, 3772, 3796, 3803, 3826, 3836, 3839, 3840, 3854, 3876, 3906, 3908,
           3914, 3916, 3940, 3962, 3974, 4011, 4027, 4058, 4098, 4100, 4129, 4212, 4230, 4236, 4237,
           4282, 4320, 4371, 4372, 4421, 4516, 4530, 4532, 4538, 4543, 4596, 4597, 4602, 4613, 4614,
           4621, 4627, 4637, 4642, 4643, 4646, 4647, 4648, 4649, 4650, 4651, 4652, 4653, 4655, 4688,
           4691, 4692, 4695, 4696, 4697, 4699, 4700, 4701, 4702, 4817, 4967, 5183, 5201, 5231, 5259,
           5263, 5267, 5295, 5328, 5330, 5354, 5411, 5414, 5420, 5448, 5469, 5560, 5698, 5831, 6064,
           6104, 6275, 6295, 6331, 6481, 6527, 6689, 6735, 6802, 6803, 6810, 6811, 6814, 6922, 6991,
           7150, 7371, 7375, 7430, 7491, 7507, 7581, 7583, 7595, 7747, 7751, 7774, 8005, 8047, 8165,
           8176, 8236, 8238, 8244, 8260, 8264, 8274, 8275, 8276, 8277, 8283, 8298, 8340, 8352, 8354,
           8355, 8356, 8357, 8358, 8360, 8361, 8362, 8365, 8372, 8404, 8420, 8436, 8577, 8578, 8584,
           8585, 8586, 8610, 8724, 8795, 8807, 8809, 8916, 8938, 9060, 9061, 9066, 9076, 9078, 9092,
           9098, 9140, 9146, 9179, 9216, 9217, 9222, 9253, 9259, 9262, 9269, 9275, 9286, 9288, 9291,
           9307, 9323, 9349, 9354, 9357, 9361, 9533, 9598, 9703, 9706, 9769, 9794, 9795, 9798, 9802,
           9806, 9807, 9815, 9818, 9819, 9823, 9825, 9829, 9831, 9836, 9837, 9849, 9884, 9886, 9892,
           9927, 9940, 9941, 9942, 9943, 9945, 9951, 10073, 10121, 10123, 10125, 10134, 10139, 10154, 10164,
           10196, 10203, 10267, 10325, 10326, 10331, 10357, 10363, 10451, 10474, 10596, 10609, 10706, 10707, 10723,
           10760, 10852, 10854, 10855, 10857, 10858, 10859, 10899, 10910, 10921, 10980, 10986, 10999, 11043, 11044,
           11052, 11091, 11241, 11286, 11363, 11589, 11738, 11867, 12052, 12068, 12134, 12142, 12216, 12287, 12317,
           12384, 12592, 12595, 12601, 12602, 12604, 12605, 12606, 12610, 12613, 12614, 12616, 12617, 12618, 12619,
           12621, 12622, 12801, 12802, 12805, 12806, 12809, 12813, 12831, 12864, 12865, 12867, 12869, 12870, 12871,
           12874, 12875, 12876, 12877, 12878, 12879, 12913, 12914, 12916, 12917, 12918, 12919, 12920, 12921, 12922,
           12923, 12924, 12925, 12926, 12927, 13049, 13053, 13055, 13059, 13064, 13068, 13126, 13172, 13193, 13236,
           13284, 13292, 13456, 13457, 13461, 13462, 13463, 13464, 13466, 13467, 13470, 13471, 13508, 13514, 13556,
           13562, 13761, 13920, 13931, 13943, 13953, 13984, 13985, 13988, 13989, 13990, 13991, 13994, 13995, 13996,
           13997, 13998, 13999, 14042, 14081, 14145, 14153, 14157, 14191, 14193, 14232, 14234, 14236, 14239, 14248,
           14249, 14313, 14320, 14325, 14374, 14378, 14381, 14485, 14498, 14534, 14543, 14557, 14595, 14625, 14628,
           14632, 14636, 14639, 14641, 14644, 14649, 14678, 14679, 14706, 14707, 14709, 14710, 14712, 14714, 14715,
           14716, 14719, 14772, 14832, 14862, 14863, 14867, 14874, 14877, 14880, 14882, 14895, 14977, 14993, 15012,
           15094, 15098, 15156, 15201, 15202, 15204, 15207, 15208, 15209, 15210, 15211, 15212, 15213, 15214, 15255,
           15305, 15307, 15314, 15349, 15456, 15458, 15460, 15461, 15462, 15463, 15464, 15465, 15466, 15468, 15469,
           15480, 15483, 15489, 15505, 15506, 15507, 15510, 15511, 15513, 15514, 15515, 15518, 15519, 15563, 15641,
           15646, 15653, 15690, 15707, 15719, 15776, 15777, 15785, 15797, 15821, 15859, 15860, 15863, 15869, 15870,
           15871, 15872, 15874, 15904, 15906, 15912, 15916, 15920, 15921, 15923, 15926, 15927, 15928, 15929, 15933,
           15958, 15965, 16033, 16073, 16083, 16215, 16268, 16304, 16305, 16306, 16307, 16308, 16309, 16311, 16312,
           16313, 16316, 16318, 16324, 16349, 16413, 16468, 16469, 16477, 16480, 16481, 16482, 16483, 16484, 16485,
           16488, 16489, 16491, 16492, 16493, 16495, 16500, 16501, 16502, 16503, 16507, 16508, 16509, 16529, 16531,
           16532, 16533, 16534, 16535, 16536, 16538, 16540, 16541, 16542, 16543, 16576, 16577, 16578, 16585, 16586,
           16610, 16616, 16620, 16634, 16741, 16784, 16785, 16789, 16886, 16887, 16888, 16913, 16928, 16971, 16977,
           17129, 17203, 17206, 17207, 17211, 17214, 17280, 17300, 17408, 17413, 17504, 17531, 17595, 17597, 17636,
           17651, 17652, 17654, 17655, 17660, 17662, 17663 };
}

std::vector<unsigned short> getWarmChannelsLHC18m_block1()
{
  return { 35, 68, 145, 192, 385, 388, 389, 390, 391, 392, 393, 394, 395, 396, 398,
           432, 433, 435, 436, 438, 439, 440, 441, 442, 445, 446, 594, 597, 598, 599,
           862, 1054, 1056, 1057, 1391, 1857, 1859, 1861, 1863, 1865, 1869, 1871, 1876, 1968, 2190,
           2209, 2211, 2213, 2215, 2217, 2219, 2221, 2223, 2244, 2350, 2421, 2542, 2678, 2688, 2793,
           2795, 2880, 2904, 3394, 3413, 3485, 3759, 4107, 4414, 4635, 4753, 5470, 5714, 5767, 6049,
           6094, 6141, 6673, 6720, 6721, 6808, 6813, 7056, 7101, 7102, 7104, 7105, 7148, 7390, 7871,
           8222, 8359, 8638, 8686, 9120, 9408, 9799, 9803, 9805, 10201, 10206, 10322, 10560, 10565, 10572,
           10657, 10658, 10659, 10660, 10661, 10662, 10663, 10664, 10665, 10666, 10667, 10668, 10670, 10671, 10702,
           10703, 10704, 10705, 10708, 10709, 10714, 10716, 10717, 10719, 10752, 10759, 10779, 10814, 10815, 10851,
           10853, 10856, 10862, 10863, 10896, 10898, 10902, 10904, 10906, 10908, 10909, 10911, 11040, 11042, 11048,
           11050, 11051, 11054, 11089, 11090, 11093, 11095, 11096, 11097, 11099, 11102, 11103, 11132, 11135, 11138,
           11141, 11148, 11150, 11197, 11198, 11280, 11329, 11380, 11553, 11630, 11647, 11937, 12260, 12270, 12276,
           12385, 12478, 12611, 12615, 12623, 12872, 12912, 12988, 13056, 13058, 13066, 13067, 13281, 13290, 13294,
           13383, 13433, 13458, 13460, 13465, 13469, 13554, 13871, 13921, 13941, 13944, 13971, 13986, 14060, 14107,
           14225, 14229, 14231, 14495, 14686, 14704, 14708, 14717, 15120, 15269, 15309, 15315, 15322, 15323, 15450,
           15749, 15753, 15758, 15856, 15857, 15858, 15861, 15862, 15864, 15865, 15866, 15867, 15868, 15905, 15907,
           15908, 15909, 15910, 15911, 15913, 15914, 15915, 15917, 15918, 15919, 15967, 16096, 16127, 16317, 16365,
           16471, 16479, 16487, 16490, 16497, 16667, 16686, 16897, 16961, 17061, 17117, 17151, 17265, 17376, 17457,
           17501, 17593, 17631 };
}

std::vector<unsigned short> getDeadChannelsLHC18m_block1()
{
  return { 1534, 2047, 2112, 2113, 2114, 2115, 2116, 2117, 2118, 2119, 2120, 2121, 2122, 2123, 2124,
           2125, 2776, 4640, 4641, 4644, 4645, 4654, 4689, 4690, 4693, 4694, 4698, 4703, 8353, 8505,
           8576, 8579, 8580, 8581, 8582, 8583, 8587, 8588, 8589, 8590, 8591, 8743, 9282, 9828, 11462,
           12147, 12593, 12594, 12596, 12597, 12598, 12599, 12600, 12603, 12607, 12800, 12803, 12804, 12807, 12808,
           12810, 12811, 12812, 12814, 12815, 12866, 12868, 12873, 12915, 13152, 13153, 13154, 13155, 13156, 13157,
           13158, 13159, 13160, 13161, 13162, 13163, 13164, 13165, 13166, 13167, 13200, 13201, 13202, 13203, 13204,
           13205, 13206, 13207, 13208, 13209, 13210, 13211, 13212, 13213, 13214, 13215, 13524, 13589, 13848, 13969,
           14443, 14624, 14626, 14627, 14629, 14630, 14631, 14633, 14634, 14635, 14637, 14638, 15051, 15200, 15203,
           15205, 15206, 15215, 15217, 15457, 15459, 15467, 15470, 15471, 15504, 15508, 15509, 15512, 15516, 15517,
           15624, 15963, 16395, 16528, 16530, 16537, 16539, 16579, 16580, 16581, 16582, 16583, 16584, 16587, 16588,
           16589, 16590, 16591, 16608, 16609, 16611, 16612, 16613, 16614, 16615, 16617, 16618, 16619, 16621, 16622,
           16623, 17200, 17201, 17202, 17204, 17205, 17208, 17209, 17210, 17212, 17213, 17215, 17648, 17649, 17650,
           17653, 17656, 17657, 17658, 17659, 17661 };
}

std::vector<unsigned short> getBadChannelsLHC17o_block2()
{
  return { 74, 103, 128, 152, 176, 191, 198, 287, 324, 353, 447, 554, 594, 655, 720,
           759, 764, 917, 1002, 1038, 1050, 1143, 1204, 1222, 1275, 1288, 1329, 1366, 1376, 1380,
           1382, 1384, 1386, 1519, 1534, 1535, 1696, 1704, 1711, 1738, 1836, 1837, 1838, 1844, 1860,
           1892, 1961, 1963, 1967, 2014, 2020, 2022, 2026, 2094, 2113, 2118, 2119, 2122, 2124, 2125,
           2126, 2127, 2193, 2196, 2210, 2245, 2298, 2325, 2395, 2397, 2399, 2406, 2424, 2474, 2487,
           2505, 2506, 2533, 2534, 2540, 2575, 2581, 2586, 2624, 2665, 2688, 2787, 2793, 2797, 2805,
           2824, 2825, 2841, 2857, 2884, 2888, 2891, 2915, 2921, 2985, 3002, 3039, 3051, 3135, 3161,
           3176, 3196, 3223, 3236, 3244, 3259, 3297, 3307, 3339, 3353, 3397, 3403, 3488, 3493, 3503,
           3528, 3732, 3740, 3748, 3754, 3772, 3796, 3803, 3826, 3836, 3839, 3840, 3854, 3876, 3906,
           3908, 3914, 3940, 3962, 3974, 4011, 4027, 4058, 4100, 4129, 4212, 4230, 4236, 4237, 4282,
           4320, 4371, 4421, 4516, 4530, 4532, 4538, 4543, 4596, 4597, 4613, 4621, 4627, 4637, 4817,
           4967, 5183, 5201, 5217, 5220, 5225, 5229, 5263, 5270, 5271, 5276, 5278, 5279, 5354, 5448,
           5602, 5603, 5604, 5606, 5607, 5608, 5648, 5649, 5650, 5651, 5652, 5655, 5656, 5657, 5659,
           5661, 5698, 5831, 5850, 6064, 6104, 6184, 6275, 6295, 6331, 6481, 6527, 6640, 6641, 6642,
           6645, 6646, 6648, 6649, 6650, 6735, 6802, 6803, 6808, 6810, 6811, 6814, 6991, 7150, 7328,
           7331, 7332, 7333, 7334, 7335, 7336, 7337, 7338, 7339, 7340, 7342, 7343, 7371, 7375, 7380,
           7383, 7388, 7389, 7390, 7417, 7425, 7430, 7457, 7491, 7520, 7521, 7522, 7524, 7525, 7526,
           7527, 7528, 7529, 7532, 7533, 7534, 7568, 7569, 7570, 7571, 7573, 7574, 7575, 7577, 7579,
           7581, 7582, 7583, 7747, 7751, 7793, 8047, 8165, 8176, 8236, 8238, 8244, 8260, 8264, 8274,
           8275, 8283, 8340, 8352, 8354, 8356, 8372, 8404, 8420, 8436, 8576, 8577, 8584, 8585, 8587,
           8610, 8628, 8630, 8724, 8807, 8809, 8916, 8938, 9056, 9060, 9066, 9076, 9078, 9092, 9098,
           9140, 9216, 9217, 9222, 9262, 9269, 9275, 9286, 9288, 9291, 9332, 9354, 9357, 9533, 9598,
           9703, 9706, 9769, 9794, 9795, 9798, 9802, 9803, 9806, 9807, 9810, 9811, 9815, 9818, 9819,
           9823, 9849, 9927, 9940, 9941, 9942, 9943, 9945, 9951, 10121, 10125, 10139, 10164, 10196, 10203,
           10326, 10331, 10357, 10363, 10451, 10505, 10609, 10611, 10660, 10666, 10718, 10723, 10921, 10986, 11091,
           11241, 11307, 11308, 11363, 11462, 11589, 11738, 11867, 11904, 11905, 11906, 11907, 11908, 11909, 11910,
           11911, 11912, 11913, 11914, 11915, 11916, 11917, 11918, 11919, 12032, 12033, 12034, 12035, 12036, 12037,
           12038, 12039, 12042, 12043, 12044, 12045, 12046, 12047, 12052, 12068, 12142, 12160, 12161, 12162, 12163,
           12164, 12165, 12166, 12167, 12168, 12169, 12170, 12171, 12172, 12173, 12174, 12175, 12216, 12311, 12317,
           12384, 12592, 12593, 12594, 12595, 12596, 12597, 12598, 12599, 12600, 12601, 12602, 12603, 12605, 12606,
           12607, 12616, 12617, 12618, 12619, 12621, 12622, 12640, 12641, 12642, 12643, 12644, 12645, 12646, 12647,
           12648, 12649, 12650, 12651, 12652, 12653, 12654, 12655, 12800, 12803, 12804, 12805, 12806, 12808, 12810,
           12812, 12813, 12815, 12831, 12848, 12851, 12852, 12855, 12859, 12860, 12861, 12862, 12863, 12864, 12865,
           12866, 12867, 12868, 12869, 12871, 12872, 12873, 12876, 12912, 12915, 12918, 12919, 12920, 12921, 12925,
           12926, 12927, 13049, 13053, 13055, 13068, 13126, 13172, 13236, 13406, 13459, 13464, 13466, 13467, 13470,
           13508, 13514, 13554, 13556, 13558, 13589, 13849, 13941, 13943, 13951, 13953, 13969, 13988, 13989, 13990,
           13991, 13994, 13995, 13996, 13997, 13998, 13999, 14042, 14049, 14059, 14061, 14063, 14081, 14108, 14111,
           14153, 14157, 14191, 14193, 14228, 14229, 14232, 14234, 14236, 14239, 14248, 14249, 14313, 14374, 14378,
           14381, 14400, 14401, 14402, 14403, 14405, 14407, 14409, 14410, 14411, 14413, 14414, 14415, 14498, 14543,
           14557, 14594, 14624, 14625, 14632, 14633, 14636, 14678, 14679, 14704, 14706, 14707, 14709, 14710, 14711,
           14712, 14714, 14715, 14716, 14718, 14719, 14772, 14832, 14862, 14867, 14874, 14895, 14928, 14977, 15012,
           15025, 15027, 15094, 15098, 15111, 15152, 15153, 15201, 15204, 15206, 15208, 15209, 15210, 15212, 15215,
           15255, 15297, 15299, 15301, 15305, 15309, 15349, 15445, 15458, 15460, 15461, 15462, 15465, 15466, 15469,
           15480, 15483, 15489, 15507, 15508, 15509, 15510, 15511, 15513, 15514, 15515, 15516, 15517, 15518, 15519,
           15653, 15690, 15719, 15754, 15776, 15785, 15798, 15799, 15800, 15802, 15804, 15805, 15806, 15807, 15821,
           15872, 15933, 15941, 15958, 15963, 15965, 15967, 16033, 16073, 16268, 16289, 16305, 16309, 16311, 16318,
           16319, 16349, 16353, 16354, 16355, 16361, 16362, 16363, 16364, 16366, 16367, 16393, 16394, 16395, 16400,
           16401, 16404, 16405, 16406, 16411, 16412, 16413, 16448, 16449, 16451, 16452, 16453, 16454, 16456, 16457,
           16459, 16460, 16461, 16463, 16507, 16510, 16529, 16532, 16534, 16535, 16536, 16540, 16541, 16543, 16576,
           16578, 16579, 16580, 16581, 16582, 16583, 16585, 16586, 16587, 16588, 16589, 16590, 16591, 16609, 16612,
           16615, 16616, 16620, 16634, 16784, 16785, 16800, 16801, 16803, 16804, 16807, 16808, 16809, 16849, 16850,
           16851, 16852, 16853, 16854, 16855, 16856, 16857, 16858, 16860, 16861, 16862, 16863, 16886, 16887, 16977,
           17093, 17117, 17129, 17184, 17187, 17189, 17191, 17197, 17200, 17201, 17202, 17203, 17210, 17211, 17214,
           17280, 17300, 17408, 17413, 17504, 17597, 17632, 17633, 17634, 17635, 17637, 17638, 17640, 17642, 17644,
           17645, 17646, 17647, 17648, 17649, 17650, 17651, 17652, 17653, 17654, 17655, 17657, 17658, 17659, 17660,
           17661, 17662, 17663 };
}

std::vector<unsigned short> getWarmChannelsLHC17o_block2()
{
  return { 68, 904, 907, 909, 947, 954, 1061, 1276, 1414, 2153, 2190, 2778, 2795, 4026, 4531,
           4986, 5600, 5601, 5605, 5609, 5610, 5611, 5612, 5613, 5615, 5653, 5654, 5658, 5660, 5714,
           5897, 6653, 6937, 7377, 7381, 8625, 8632, 8813, 9202, 9799, 9801, 9805, 9820, 9821, 9875,
           10123, 10565, 10779, 10831, 11042, 11044, 11048, 11050, 11052, 11093, 11095, 11102, 11139, 11141, 11148,
           11197, 11243, 11411, 11630, 11647, 12040, 12041, 12276, 12308, 12309, 12310, 12314, 12316, 12319, 12430,
           12610, 12611, 12870, 12874, 12988, 13281, 13282, 13286, 13289, 13291, 13292, 13293, 13383, 13402, 13403,
           13407, 13462, 13463, 13465, 13471, 13562, 13761, 13903, 13985, 14057, 14060, 14373, 14485, 14600, 14708,
           14717, 14789, 14839, 14987, 14993, 15054, 15108, 15110, 15118, 15119, 15162, 15165, 15167, 15296, 15298,
           15304, 15307, 15401, 15749, 15758, 16308, 16310, 16315, 16317, 16358, 16365, 16389, 16392, 16402, 16409,
           16425, 17396, 17452, 17463, 17537 };
}

std::vector<unsigned short> getDeadChannelsLHC17o_block2()
{
  return { 2047, 2112, 2114, 2115, 2116, 2117, 2120, 2123, 2776, 5216, 5218, 5219, 5221, 5222, 5223,
           5224, 5226, 5227, 5228, 5230, 5231, 5264, 5265, 5266, 5267, 5268, 5269, 5272, 5273, 5274,
           5275, 5277, 5280, 5281, 5282, 5283, 5284, 5285, 5286, 5287, 5288, 5289, 5290, 5291, 5292,
           5293, 5294, 5295, 5328, 5329, 5330, 5331, 5332, 5333, 5334, 5335, 5336, 5337, 5338, 5339,
           5340, 5341, 5342, 5343, 7330, 7341, 7523, 7530, 7531, 7535, 7572, 7576, 7578, 7580, 8353,
           8578, 8579, 8580, 8581, 8582, 8583, 8586, 8588, 8589, 8590, 8591, 8743, 8832, 8833, 8834,
           8835, 8836, 8837, 8838, 8839, 8840, 8841, 8842, 8843, 8844, 8845, 8846, 8847, 8880, 8881,
           8882, 8883, 8884, 8885, 8886, 8887, 8888, 8889, 8890, 8891, 8892, 8893, 8894, 8895, 9282,
           9361, 12147, 12604, 12801, 12802, 12807, 12809, 12811, 12814, 12849, 12850, 12853, 12854, 12856, 12857,
           12858, 12913, 12914, 12916, 12917, 12922, 12923, 12924, 13524, 13848, 14406, 14443, 14626, 14627, 14628,
           14629, 14630, 14631, 14634, 14635, 14637, 14638, 14639, 15051, 15200, 15202, 15203, 15205, 15207, 15211,
           15213, 15214, 15217, 15456, 15457, 15459, 15463, 15464, 15467, 15468, 15470, 15471, 15504, 15505, 15506,
           15512, 15624, 16450, 16455, 16458, 16462, 16528, 16530, 16531, 16533, 16537, 16538, 16539, 16542, 16577,
           16584, 16608, 16610, 16611, 16613, 16614, 16617, 16618, 16619, 16621, 16622, 16623, 16802, 16805, 16806,
           16810, 16811, 16812, 16813, 16814, 16815, 16848, 16859, 17185, 17186, 17188, 17190, 17192, 17193, 17194,
           17195, 17196, 17198, 17199, 17204, 17205, 17206, 17207, 17208, 17209, 17212, 17213, 17215, 17636, 17639,
           17641, 17643, 17656

  };
}

} // namespace emcal

} // namespace o2