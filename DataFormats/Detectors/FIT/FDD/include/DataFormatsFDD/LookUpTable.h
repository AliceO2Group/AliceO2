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
#ifndef ALICEO2_FDD_LOOKUPTABLE_H_
#define ALICEO2_FDD_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FDD
//////////////////////////////////////////////

#include <Rtypes.h>
#include <cassert>
#include <iostream>
#include <iomanip> // std::setfill, std::setw - for stream formating
#include <Framework/Logger.h>
#include "FDDBase/Constants.h"

namespace o2
{
namespace fdd
{

struct Topo {
  int modLink = 0; // Number of Processing Module, associated with GBT link ID
  int modCh = 0;   // Channel within the Processing Module in range from 0-11
  ClassDefNV(Topo, 1);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  return (a.modLink < b.modLink || (a.modLink == b.modLink && a.modCh < b.modCh));
}

class LookUpTable
{
 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  LookUpTable() = default;
  ~LookUpTable() = default;

  explicit LookUpTable(bool fillLinearly)
    : mTopoVector(Nmodules * NChPerMod, {0, 0}),
      mInvTopo(mTopoVector.size(), 0)
  {
    if (fillLinearly) {
      LOG(INFO) << "Mapping of global channel and (PM, PM channel) pair";
      for (int link = 0; link < Nmodules; ++link) {
        for (int ch = 0; ch < NChPerMod; ++ch) {
          mTopoVector[link * NChPerMod + ch] = o2::fdd::Topo{link, ch};
        }
      }
    } else {
      // TODO: If needed: implement more realistic splitting: 1 ring -> 1 PM instead of linear
      LOG(WARNING) << "Don't use it - not implemented yet.";
    }

    // Fill inverted LUT - matters only if LUT is not linear
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo[getIdx(mTopoVector[channel].modLink, mTopoVector[channel].modCh)] = channel;
    }
  }

  int getChannel(int link, int mcp) const { return mInvTopo[getIdx(link, mcp)]; }
  int getLink(int channel) const { return mTopoVector[channel].modLink; }
  int getModChannel(int channel) const { return mTopoVector[channel].modCh; }
  int getTcmLink() const { return Nmodules; }
  void printFullMap() const
  {
    std::cout << "o2::fdd::LookUpTable::printFullMap(): mTopoVector: [globalCh  link  modCh]" << std::endl;
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      std::cout << "  " << std::right << std::setw(2) << channel << "  ";
      std::cout << std::right << std::setw(2) << mTopoVector[channel].modLink << "  ";
      std::cout << std::right << std::setw(3) << mTopoVector[channel].modCh << std::endl;
    }
    std::cout << "o2::fdd::LookUpTable::printFullMap(): mInvTopo: [idx  globalCh    link  modCh]" << std::endl;
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx) {
      std::cout << "  " << std::right << std::setw(3) << idx << "  ";
      std::cout << std::right << std::setw(3) << mInvTopo[idx] << "    ";
      std::cout << std::right << std::setw(2) << getLinkFromIdx(mInvTopo[idx]) << "  ";
      std::cout << std::right << std::setw(2) << getModChannelFromIdx(mInvTopo[idx]) << std::endl;
    }
  }

  static o2::fdd::LookUpTable linear()
  {
    return o2::fdd::LookUpTable{1};
  }

 private:
  std::vector<Topo> mTopoVector; // iterator of each vector element gives the global channel number
  std::vector<int> mInvTopo;     // each element is an iterator of mTopoVector

  static int getIdx(int link, int modCh)
  {
    assert(modCh < NChPerMod);
    return link * NChPerMod + modCh;
  }
  static int getLinkFromIdx(int idx) { return idx / NChPerMod; }
  static int getModChannelFromIdx(int idx) { return idx % NChPerMod; }

  ClassDefNV(LookUpTable, 1);
};

} // namespace fdd
} // namespace o2
#endif
