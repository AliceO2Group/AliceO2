// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
//file RawEventData.h class  for RAW data format
// Alla.Maevskaya
//  simple look-up table just to feed digits 2 raw procedure.
//Will be really set after module/electronics connections
//
#ifndef ALICEO2_FV0_LOOKUPTABLE_H_
#define ALICEO2_FV0_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FV0
//////////////////////////////////////////////

#include <Rtypes.h>
#include <cassert>
#include <iostream>
#include <tuple>
#include <Framework/Logger.h>

namespace o2
{
namespace fv0
{
struct Topo {
  int pmN = 0;
  int pmCh = 0;
  ClassDefNV(Topo, 1);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  return (a.pmN < b.pmN || (a.pmN == b.pmN && a.pmCh < b.pmCh));
}

class LookUpTable
{
  static constexpr int sNumberOfChannelsPerPm = 12;
  static constexpr int sNumberOfPms = 4;

 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  LookUpTable() = default;

  explicit LookUpTable(std::vector<Topo> const& topoVector)
    : mTopoVector(topoVector), mInvTopo(topoVector.size())
  {
    LOG(INFO) << "Mapping of channel and PMT channel [Const: LookUptable]";
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo.at(getIdx(mTopoVector[channel].pmN, mTopoVector[channel].pmN)) = channel;
    }
  }
  ~LookUpTable() = default;
  void printFullMap() const
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      std::cout << channel << "\t :  PM \t" << mTopoVector[channel].pmN << " MCP \t" << mTopoVector[channel].pmN << std::endl;
    }
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx) {
      std::cout << "PM \t" << getLinkFromIdx(mInvTopo[idx]) << " MCP \t" << getPmChannelFromIdx(mInvTopo[idx]) << std::endl;
    }
  }

  int getChannel(int link, int mcp) const
  {
    return mInvTopo[getIdx(link, mcp)];
  }

  int getLink(int channel) const { return mTopoVector[channel].pmN; }
  int getPmChannel(int channel) const { return mTopoVector[channel].pmCh; }

 private:
  std::vector<Topo> mTopoVector;
  std::vector<int> mInvTopo;

  static int getIdx(int link, int pmCh)
  {
    assert(pmCh < sNumberOfChannelsPerPm);
    return link * sNumberOfChannelsPerPm + pmCh;
  }
  static int getLinkFromIdx(int idx)
  {
    return idx / sNumberOfChannelsPerPm;
  }
  static int getPmChannelFromIdx(int idx)
  {
    return idx % sNumberOfChannelsPerPm;
  }

  ClassDefNV(LookUpTable, 1);
};

} // namespace fv0
} // namespace o2
#endif
