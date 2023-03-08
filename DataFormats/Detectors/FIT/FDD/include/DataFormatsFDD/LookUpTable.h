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
//
// file RawEventData.h class  for RAW data format
// Alla.Maevskaya
//  simple look-up table just to feed digits 2 raw procedure.
// Will be really set after module/electronics connections
//
#ifndef ALICEO2_FDD_LOOKUPTABLE_H_
#define ALICEO2_FDD_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FDD
//////////////////////////////////////////////

#include "DataFormatsFIT/LookUpTable.h"
#include <Rtypes.h>
#include <cassert>
#include <iostream>
#include <iomanip> // std::setfill, std::setw - for stream formating
#include <Framework/Logger.h>
#include "FDDBase/Constants.h"
#include "CommonUtils/NameConf.h"

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
      LOG(info) << "Mapping of global channel and (PM, PM channel) pair";
      for (int link = 0; link < Nmodules; ++link) {
        for (int ch = 0; ch < NChPerMod; ++ch) {
          mTopoVector[link * NChPerMod + ch] = o2::fdd::Topo{link, ch};
        }
      }
    } else {
      // TODO: If needed: implement more realistic splitting: 1 ring -> 1 PM instead of linear
      LOG(warning) << "Don't use it - not implemented yet.";
    }

    // Fill inverted LUT - matters only if LUT is not linear
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo[getIdx(mTopoVector[channel].modLink, mTopoVector[channel].modCh)] = channel;
    }
  }

  int getChannel(int link, int mcp, int ep = 0) const { return mInvTopo[getIdx(link, mcp)]; }
  int getLink(int channel) const { return mTopoVector[channel].modLink; }
  int getModChannel(int channel) const { return mTopoVector[channel].modCh; }
  int getTcmLink() const { return Nmodules; }
  bool isTCM(int link, int ep) const { return link == 2 && ep == 0; }
  Topo getTopoPM(int globalChannelID) const { return mTopoVector[globalChannelID]; }
  Topo getTopoTCM() const { return Topo{getTcmLink(), 0}; }
  std::size_t getNchannels() const { return mTopoVector.size(); } // get number of global PM channels
  void printFullMap() const
  {
    LOG(info) << "o2::fdd::LookUpTable::printFullMap(): mTopoVector: [globalCh  link  modCh]";
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      LOG(info) << "  " << channel << "  " << mTopoVector[channel].modLink << "  " << mTopoVector[channel].modCh;
    }
    LOG(info) << "o2::fdd::LookUpTable::printFullMap(): mInvTopo: [idx  globalCh    link  modCh]";
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx) {
      LOG(info) << "  " << idx << "  " << mInvTopo[idx] << "    " << getLinkFromIdx(mInvTopo[idx]) << "  " << getModChannelFromIdx(mInvTopo[idx]);
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

namespace deprecated
{
// Singleton for LookUpTable
class SingleLUT : public LookUpTable
{
 private:
  SingleLUT() : LookUpTable(LookUpTable::linear()) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;

 public:
  typedef Topo Topo_t;
  static constexpr char sDetectorName[] = "FDD";
  static SingleLUT& Instance()
  {
    static SingleLUT instanceLUT;
    return instanceLUT;
  }
  // Temporary
  // Making topo for FEE recognizing(Local channelID is supressed)
  static Topo_t makeGlobalTopo(const Topo_t& topo)
  {
    return Topo_t{topo.modLink, 0};
  }
  static int getLocalChannelID(const Topo_t& topo)
  {
    return topo.modCh;
  }
  // Prepare full map for FEE metadata
  template <typename RDHtype, typename RDHhelper = void>
  auto makeMapFEEmetadata() -> std::map<Topo_t, RDHtype>
  {
    std::map<Topo_t, RDHtype> mapResult;
    const uint16_t cruID = 0;      // constant
    const uint32_t endPointID = 0; // constant
    uint64_t feeID = 0;            // increments
    // PM
    for (int iCh = 0; iCh < Instance().getNchannels(); iCh++) {
      auto pairInserted = mapResult.insert({makeGlobalTopo(Instance().getTopoPM(iCh)), RDHtype{}});
      if (pairInserted.second) {
        auto& rdhObj = pairInserted.first->second;
        const auto& topoObj = pairInserted.first->first;
        if constexpr (std::is_same<RDHhelper, void>::value) {
          rdhObj.linkID = topoObj.modLink;
          rdhObj.endPointID = endPointID;
          rdhObj.feeId = feeID;
          rdhObj.cruID = cruID;
        } else // Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.modLink);
          RDHhelper::setEndPointID(&rdhObj, endPointID);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
        feeID++;
      }
    }
    // TCM
    {
      auto pairInserted = mapResult.insert({makeGlobalTopo(Instance().getTopoTCM()), RDHtype{}});
      if (pairInserted.second) {
        auto& rdhObj = pairInserted.first->second;
        const auto& topoObj = pairInserted.first->first;
        if constexpr (std::is_same<RDHhelper, void>::value) {
          rdhObj.linkID = topoObj.modLink;
          rdhObj.endPointID = endPointID;
          rdhObj.feeId = feeID;
          rdhObj.cruID = cruID;
        } else // Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.modLink);
          RDHhelper::setEndPointID(&rdhObj, endPointID);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
      } else {
        LOG(info) << "WARNING! CHECK LUT! TCM METADATA IS INCORRECT!";
      }
    }
    assert(mapResult.size() > 0);
    return mapResult;
  }
};
} // namespace deprecated
namespace new_lut
{
// Singleton for LookUpTable
template <typename LUT>
class SingleLUT : public LUT
{
 private:
  SingleLUT(const std::string& ccdbPath, const std::string& ccdbPathToLUT) : LUT(ccdbPath, ccdbPathToLUT) {}
  SingleLUT(const std::string& pathToFile) : LUT(pathToFile) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;

 public:
  static constexpr char sDetectorName[] = "FDD";
  static constexpr char sDefaultLUTpath[] = "FDD/Config/LookupTable";
  inline static std::string sCurrentCCDBpath = "";
  inline static std::string sCurrentLUTpath = sDefaultLUTpath;
  // Before instance() call, setup url and path
  static void setCCDBurl(const std::string& url) { sCurrentCCDBpath = url; }
  static void setLUTpath(const std::string& path) { sCurrentLUTpath = path; }
  static SingleLUT& Instance()
  {
    if (sCurrentCCDBpath == "") {
      sCurrentCCDBpath = o2::base::NameConf::getCCDBServer();
    }
    static SingleLUT instanceLUT(sCurrentCCDBpath, sCurrentLUTpath);
    return instanceLUT;
  }
};
} // namespace new_lut

using SingleLUT = new_lut::SingleLUT<o2::fit::LookupTableBase<>>;

} // namespace fdd
} // namespace o2
#endif
