// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include <map>
#include <fmt/format.h>
#include "MCHMappingInterface/Segmentation.h"
#include "o2mchmappingimpl4_export.h"
namespace
{
const std::map<int, int> deId2SameType{
  {100, 100},
  {101, 100},
  {102, 100},
  {103, 100},
  {200, 100},
  {201, 100},
  {202, 100},
  {203, 100},
  {300, 300},
  {301, 300},
  {302, 300},
  {303, 300},
  {400, 300},
  {401, 300},
  {402, 300},
  {403, 300},
  {500, 500},
  {501, 501},
  {502, 502},
  {503, 503},
  {504, 504},
  {505, 504},
  {506, 503},
  {507, 502},
  {508, 501},
  {509, 500},
  {510, 501},
  {511, 502},
  {512, 503},
  {513, 504},
  {514, 504},
  {515, 503},
  {516, 502},
  {517, 501},
  {600, 600},
  {601, 601},
  {602, 602},
  {603, 503},
  {604, 504},
  {605, 504},
  {606, 503},
  {607, 602},
  {608, 601},
  {609, 600},
  {610, 601},
  {611, 602},
  {612, 503},
  {613, 504},
  {614, 504},
  {615, 503},
  {616, 602},
  {617, 601},
  {700, 700},
  {701, 701},
  {702, 702},
  {703, 703},
  {704, 704},
  {705, 705},
  {706, 706},
  {707, 706},
  {708, 705},
  {709, 704},
  {710, 703},
  {711, 702},
  {712, 701},
  {713, 700},
  {714, 701},
  {715, 702},
  {716, 703},
  {717, 704},
  {718, 705},
  {719, 706},
  {720, 706},
  {721, 705},
  {722, 704},
  {723, 703},
  {724, 702},
  {725, 701},
  {800, 700},
  {801, 701},
  {802, 702},
  {803, 703},
  {804, 704},
  {805, 705},
  {806, 706},
  {807, 706},
  {808, 705},
  {809, 704},
  {810, 703},
  {811, 702},
  {812, 701},
  {813, 700},
  {814, 701},
  {815, 702},
  {816, 703},
  {817, 704},
  {818, 705},
  {819, 706},
  {820, 706},
  {821, 705},
  {822, 704},
  {823, 703},
  {824, 702},
  {825, 701},
  {900, 700},
  {901, 701},
  {902, 902},
  {903, 903},
  {904, 904},
  {905, 905},
  {906, 705},
  {907, 705},
  {908, 905},
  {909, 904},
  {910, 903},
  {911, 902},
  {912, 701},
  {913, 700},
  {914, 701},
  {915, 902},
  {916, 903},
  {917, 904},
  {918, 905},
  {919, 705},
  {920, 705},
  {921, 905},
  {922, 904},
  {923, 903},
  {924, 902},
  {925, 701},
  {1000, 700},
  {1001, 701},
  {1002, 902},
  {1003, 903},
  {1004, 904},
  {1005, 905},
  {1006, 705},
  {1007, 705},
  {1008, 905},
  {1009, 904},
  {1010, 903},
  {1011, 902},
  {1012, 701},
  {1013, 700},
  {1014, 701},
  {1015, 902},
  {1016, 903},
  {1017, 904},
  {1018, 905},
  {1019, 705},
  {1020, 705},
  {1021, 905},
  {1022, 904},
  {1023, 903},
  {1024, 902},
  {1025, 701}};

} // namespace

namespace
{
std::map<int, o2::mch::mapping::Segmentation*> createSegmentations()
{
  std::map<int, o2::mch::mapping::Segmentation*> segs;
  for (auto deid : {100,
                    300,
                    500,
                    501,
                    502,
                    503,
                    504,
                    600,
                    601,
                    602,
                    700,
                    701,
                    702,
                    703,
                    704,
                    705,
                    706,
                    902,
                    903,
                    904,
                    905}) {
    segs.emplace(deid, new o2::mch::mapping::Segmentation(deid));
  };
  return segs;
} // namespace

} // namespace

namespace o2::mch::mapping
{

O2MCHMAPPINGIMPL4_EXPORT
const Segmentation& segmentation(int detElemId)
{
  static auto segs = createSegmentations();
  auto refDeForThatSegmentation = deId2SameType.find(detElemId);
  return *(segs[refDeForThatSegmentation->second]);
}
} // namespace o2::mch::mapping
