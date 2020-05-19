// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "MCHRawCommon/SampaCluster.h"
#include "MCHRawCommon/DataFormats.h"
#include "Assertions.h"
#include <fmt/format.h>
#include <sstream>

namespace o2
{
namespace mch
{
namespace raw
{

SampaCluster::SampaCluster(uint10_t sampaTime, uint20_t bunchCrossing, uint20_t chargeSum)
  : sampaTime(impl::assertIsInRange("sampaTime", sampaTime, 0, 0x3FF)),
    bunchCrossing(impl::assertIsInRange("bunchCrossing", bunchCrossing, 0, 0xFFFFF)),
    chargeSum(impl::assertIsInRange("chargeSum", chargeSum, 0, 0xFFFFF)), // 20 bits
    samples{}

{
}

SampaCluster::SampaCluster(uint10_t sampaTime, uint20_t bunchCrossing, const std::vector<uint10_t>& samples)
  : sampaTime(impl::assertIsInRange("sampaTime", sampaTime, 0, 0x3FF)),
    bunchCrossing(impl::assertIsInRange("bunchCrossing", bunchCrossing, 0, 0xFFFFF)),
    chargeSum(0),
    samples(samples.begin(), samples.end())
{
  if (samples.empty()) {
    throw std::invalid_argument("cannot add data with no sample");
  }
  for (auto i = 0; i < samples.size(); i++) {
    impl::assertIsInRange("sample", samples[i], 0, 0x3FF);
  }
}

uint16_t SampaCluster::nofSamples() const
{
  if (!samples.empty()) {
    return samples.size();
  }
  return 1;
}

bool SampaCluster::isClusterSum() const
{
  return samples.empty();
}

uint16_t SampaCluster::nof10BitWords() const
{
  uint16_t n10{2}; // 10 bits (nsamples) + 10 bits (sampaTime)
  if (isClusterSum()) {
    n10 += 2; // 20 bits (chargesum)
  } else {
    for (auto s : samples) {
      ++n10; // 10 bits for each sample
    }
  }
  return n10;
}

std::ostream& operator<<(std::ostream& os, const SampaCluster& sc)
{
  os << fmt::format("ts {:4d} ", sc.sampaTime);
  os << fmt::format("bc {:4d} ", sc.bunchCrossing);
  if (sc.isClusterSum()) {
    os << fmt::format("q {:6d}", sc.chargeSum);
  } else {
    os << fmt::format("n {:4d} q [ ", sc.samples.size());
    for (auto s : sc.samples) {
      os << fmt::format("{:4d} ", s);
    }
    os << "]";
  }
  return os;
}

std::string asString(const SampaCluster& sc)
{
  std::string s = fmt::format("ts-{}-bc-{}-q", sc.sampaTime, sc.bunchCrossing);
  if (sc.isClusterSum()) {
    s += fmt::format("-{}", sc.chargeSum);
  } else {
    for (auto sample : sc.samples) {
      s += fmt::format("-{}", sample);
    }
  }
  return s;
}

} // namespace raw
} // namespace mch
} // namespace o2
