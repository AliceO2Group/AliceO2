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
#ifndef ALICEO2_FT0_LOOKUPTABLE_H_
#define ALICEO2_FT0_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FT0
//////////////////////////////////////////////

#include <Rtypes.h>
#include <cassert>
#include <iostream>
#include <tuple>
namespace o2
{
namespace ft0
{
struct Topo {
  int mPM = 0;
  int mMCP = 0;
  ClassDefNV(Topo, 1);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  return (a.mPM < b.mPM || (a.mPM == b.mPM && a.mMCP < b.mMCP));
}

class LookUpTable
{
  static constexpr int NUMBER_OF_MCPs = 12;
  static constexpr int NUMBER_OF_PMs = 18;

 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  LookUpTable() = default;
  explicit LookUpTable(std::vector<Topo> const& topoVector)
    : mTopoVector(topoVector), mInvTopo(topoVector.size())
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel)
      mInvTopo.at(getIdx(mTopoVector[channel].mPM, mTopoVector[channel].mMCP)) = channel;
  }
  ~LookUpTable() = default;
  void printFullMap() const
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel)
      std::cout << channel << "\t :  PM \t" << mTopoVector[channel].mPM << " MCP \t" << mTopoVector[channel].mMCP << std::endl;
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx)
      std::cout << "PM \t" << getLinkFromIdx(mInvTopo[idx]) << " MCP \t" << getMCPFromIdx(mInvTopo[idx]) << std::endl;
  }

  int getChannel(int link, int mcp) const
  {
    return mInvTopo[getIdx(link, mcp)];
  }

  int getLink(int channel) const { return mTopoVector[channel].mPM; }
  int getMCP(int channel) const { return mTopoVector[channel].mMCP; }

 private:
  std::vector<Topo> mTopoVector;
  std::vector<int> mInvTopo;

  static int getIdx(int link, int mcp)
  {
    assert(mcp < NUMBER_OF_MCPs);
    return link * NUMBER_OF_MCPs + mcp;
  }
  static int getLinkFromIdx(int idx)
  {
    return idx / NUMBER_OF_MCPs;
  }
  static int getMCPFromIdx(int idx)
  {
    return idx % NUMBER_OF_MCPs;
  }

  ClassDefNV(LookUpTable, 1);
};

} // namespace ft0
} // namespace o2
#endif
