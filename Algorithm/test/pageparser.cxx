// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// @file   parser.cxx
/// @author Matthias Richter
/// @since  2017-09-27
/// @brief  Unit test for parser of objects in memory pages

#define BOOST_TEST_MODULE Test Algorithm Parser
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#include <boost/test/unit_test.hpp>
#include <iostream>
#include <iomanip>
#include <vector>
#include "Headers/DataHeader.h" // hexdump
#include "../include/Algorithm/PageParser.h"
#include "StaticSequenceAllocator.h"

struct PageHeader {
  uint32_t magic = 0x45474150;
  uint32_t pageid;

  PageHeader(uint32_t id) : pageid(id) {}
};

struct ClusterData {
  uint32_t magic = 0x54534c43;
  uint32_t clusterid;
  uint16_t x;
  uint16_t y;
  uint16_t z;
  uint8_t e;

  ClusterData()
    : clusterid(0)
    , x(0)
    , y(0)
    , z(0)
    , e(0)
  {}

  ClusterData(uint32_t _id, uint16_t _x, uint16_t _y, uint16_t _z, uint8_t _e)
    : clusterid(_id)
    , x(_x)
    , y(_y)
    , z(_z)
    , e(_e)
  {}

  bool operator==(const ClusterData& rhs) {
    return clusterid == rhs.clusterid
      && x == rhs.x
      && y == rhs.y
      && z == rhs.z
      && e == rhs.e;
  }
};

template<typename ListT,
         typename PageHeaderT>
std::pair<std::unique_ptr<uint8_t>, size_t> MakeBuffer(unsigned pagesize,
                                                       PageHeaderT pageheader,
                                                       ListT& dataset)
{
  auto totalSize = dataset.size() * sizeof(typename ListT::value_type);
  unsigned nPages = 0;
  do {
    totalSize += sizeof(PageHeaderT);
    ++nPages;
  } while (nPages * pagesize < totalSize);

  std::unique_ptr<uint8_t> buffer(new uint8_t[totalSize]);

  unsigned position = 0;
  auto target = buffer.get();
  for (auto element : dataset) {
    auto source = reinterpret_cast<uint8_t*>(&element);
    auto copySize = sizeof(typename ListT::value_type);
    if ((position % pagesize) == 0) {
      memcpy(target, &pageheader, sizeof(PageHeaderT));
      position += sizeof(PageHeaderT);
      target += sizeof(PageHeaderT);
    }
    if ((position % pagesize) + copySize > pagesize) {
      copySize = ((position % pagesize) + copySize) - pagesize;
    }
    if (copySize > 0) {
      memcpy(target, source, copySize);
      position += copySize;
      target += copySize;
      source += copySize;
    }
    copySize = sizeof(typename ListT::value_type) - copySize;
    if (copySize > 0) {
      memcpy(target, &pageheader, sizeof(PageHeaderT));
      position += sizeof(PageHeaderT);
      target += sizeof(PageHeaderT);
      memcpy(target, source, copySize);
    }
    position += copySize;
    target += copySize;
  }

  std::pair<std::unique_ptr<uint8_t>, size_t> result;
  result.first = std::move(buffer);
  result.second = totalSize;
  return result;
}

template<typename ListT>
void FillData(ListT& dataset, unsigned entries)
{
  for (unsigned i = 0; i < entries; i++) {
    dataset.emplace_back(i, 0xaa, 0xbb, 0xcc, 0xd);
  }
}

BOOST_AUTO_TEST_CASE(test_pageparser)
{
  constexpr unsigned pagesize = 128;
  std::vector<ClusterData> dataset;
  FillData(dataset, 20);
  auto buffer = MakeBuffer(pagesize, PageHeader(0), dataset);
  o2::Header::hexDump("pagebuffer", buffer.first.get(), buffer.second);

  using RawParser = o2::algorithm::PageParser<PageHeader, pagesize, ClusterData>;
  RawParser parser(buffer.first.get(), buffer.second);

  unsigned dataidx = 0;
  for (auto i : parser) {
    o2::Header::hexDump("clusterdata", &i, sizeof(ClusterData));
    BOOST_REQUIRE( i == dataset[dataidx++]);
  }

  std::vector<RawParser::value_type> linearizedData;
  linearizedData.insert(linearizedData.begin(), parser.begin(), parser.end());
  dataidx = 0;
  for (auto i : linearizedData) {
    BOOST_REQUIRE( i == dataset[dataidx++]);
  }
}
