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
#include <iomanip> // std::setfill, std::setw - for stream formating
#include <Framework/Logger.h>
#include "FV0Base/Constants.h"

namespace o2
{
namespace fv0
{

struct Topo {
  int pmLink = 0; // Number of Processing Module, associated with GBT link ID
  int pmCh = 0;   // Channel within the Processing Module in range from 0-11
  ClassDefNV(Topo, 1);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  return (a.pmLink < b.pmLink || (a.pmLink == b.pmLink && a.pmCh < b.pmCh));
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
    : mTopoVector(Constants::nPms * Constants::nChannelsPerPm, {0, 0}),
      mInvTopo(mTopoVector.size(), 0)
  {
    if (fillLinearly) {
      LOG(INFO) << "Mapping of global channel and (PM, PM channel) pair";
      for (int link = 0; link < Constants::nPms; ++link) {
        for (int ch = 0; ch < Constants::nChannelsPerPm; ++ch) {
          mTopoVector[link * Constants::nChannelsPerPm + ch] = o2::fv0::Topo{link, ch};
        }
      }
    } else {
      // TODO: If needed: implement more realistic splitting: 1 ring -> 1 PM instead of linear
      LOG(WARNING) << "Don't use it - not implemented yet.";
    }

    // Fill inverted LUT - matters only if LUT is not linear
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo[getIdx(mTopoVector[channel].pmLink, mTopoVector[channel].pmCh)] = channel;
    }
  }

  int getChannel(int link, int mcp) const { return mInvTopo[getIdx(link, mcp)]; }
  int getLink(int channel) const { return mTopoVector[channel].pmLink; }
  int getPmChannel(int channel) const { return mTopoVector[channel].pmCh; }
  int getTcmLink() const { return mLinkTCM; }
  bool isTCM(int link, int ep) const { return link == mLinkTCM && ep == 0; }
  void printFullMap() const
  {
    std::cout << "o2::fv0::LookUpTable::printFullMap(): mTopoVector: [globalCh  link  pmCh]" << std::endl;
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      std::cout << "  " << std::right << std::setw(2) << channel << "  ";
      std::cout << std::right << std::setw(2) << mTopoVector[channel].pmLink << "  ";
      std::cout << std::right << std::setw(3) << mTopoVector[channel].pmCh << std::endl;
    }
    std::cout << "o2::fv0::LookUpTable::printFullMap(): mInvTopo: [idx  globalCh    link  pmCh]" << std::endl;
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx) {
      std::cout << "  " << std::right << std::setw(3) << idx << "  ";
      std::cout << std::right << std::setw(3) << mInvTopo[idx] << "    ";
      std::cout << std::right << std::setw(2) << getLinkFromIdx(mInvTopo[idx]) << "  ";
      std::cout << std::right << std::setw(2) << getPmChannelFromIdx(mInvTopo[idx]) << std::endl;
    }
  }

 private:
  std::vector<Topo> mTopoVector; // iterator of each vector element gives the global channel number
  std::vector<int> mInvTopo;     // each element is an iterator of mTopoVector
  static constexpr int mLinkTCM = Constants::nPms;

  static int getIdx(int link, int pmCh)
  {
    assert(pmCh < Constants::nChannelsPerPm);
    return link * Constants::nChannelsPerPm + pmCh;
  }
  static int getLinkFromIdx(int idx) { return idx / Constants::nChannelsPerPm; }
  static int getPmChannelFromIdx(int idx) { return idx % Constants::nChannelsPerPm; }

  ClassDefNV(LookUpTable, 1);
};

} // namespace fv0
} // namespace o2
#endif
