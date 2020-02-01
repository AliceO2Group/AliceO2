// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#define BOOST_TEST_MODULE Test DataFormatsITSMFT
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include "DataFormatsITSMFT/Cluster.h"

namespace o2::itsmft
{

// we explicitly declare o2::itsmft::Cluster as messageable and have to make
// sure that this is fulfilled
BOOST_AUTO_TEST_CASE(Cluster_messageable)
{
  using ElementType = Cluster;
  static_assert(o2::framework::is_messageable<ElementType>::value == true);
  std::vector<ElementType> clusters(10);
  auto makeElement = [](size_t idx) {
    return ElementType{};
  };
  size_t idx = 0;
  for (auto& cluster : clusters) {
    cluster.setXYZ(idx, idx + 10, idx + 20);
    idx++;
  }

  size_t memsize = sizeof(ElementType) * clusters.size();
  auto buffer = std::make_unique<char[]>(memsize);
  memcpy(buffer.get(), (char*)clusters.data(), memsize);
  auto* pclone = reinterpret_cast<ElementType*>(buffer.get());

  for (auto const& cluster : clusters) {
    BOOST_REQUIRE(cluster.getXYZ() == pclone->getXYZ());
    ++pclone;
  }
}

} // namespace o2::itsmft
