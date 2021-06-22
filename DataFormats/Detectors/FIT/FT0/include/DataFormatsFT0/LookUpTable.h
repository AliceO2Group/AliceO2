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
// file RawEventData.h class  for RAW data format
// Alla.Maevskaya
//  simple look-up table just to feed digits 2 raw procedure.
// Will be really set after module/electronics connections
//
#ifndef ALICEO2_FT0_LOOKUPTABLE_H_
#define ALICEO2_FT0_LOOKUPTABLE_H_
////////////////////////////////////////////////
// Look Up Table FT0
//////////////////////////////////////////////

#include "CCDB/BasicCCDBManager.h"

#include <Rtypes.h>
#include <cassert>
#include <exception>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <tuple>
#include <TSystem.h>
#include <cstdlib>
#include <map>
#include <string_view>
#include <vector>
#include <cstdlib>

namespace o2
{
namespace ft0
{
struct Topo {
  int mPM = 0;
  int mMCP = 0;
  int mEP = 0;
  ClassDefNV(Topo, 2);
};

// enum class Side : char { A, C };
// struct PM {
//   Side side;
//   uint8_t PM_num, PM_channel;
// };

struct HVchannel {
  /*  enum class HVBoard : uint8_t { NA,
                                 A_out,
                                 A_in,
                                 C_up,
                                 C_down,
                                 C_mid };*/

  uint8_t channel;
  Topo pm;
  std::string HV_board;
  uint8_t HV_channel;
  std::string MCP_SN;
  std::string HV_cabel;
  std::string signal_cable;
  std::string EP;

  ClassDefNV(HVchannel, 2);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  /* return (a.mPM < b.mPM || (a.mPM == b.mPM && a.mMCP < b.mMCP)); */
  auto t = [](Topo const& x) -> decltype(auto) { return std::tie(x.mPM, x.mMCP, x.mEP); };
  return t(a) < t(b);
}

inline o2::ft0::Topo read_Topo(std::string_view str)
{
  assert(str.substr(0, 2) == "PM"); // && str[4] == '/' && str[5] == 'C' && str[6] == 'h');
  char side = str[2];
  char* ptr;
  uint8_t pm_num = std::strtol(str.data() + 3, &ptr, 10); // = str[3] - '0';
  /* auto res = std::from_chars(str.data()+3, str.data()+3+str.size(), pm_num); */
  /* if (res.ec != std::errc() || res.ptr[0] != '/') */
  if (errno || ptr[0] != '/') {
    throw std::invalid_argument("Cannot read pm_num");
  }
  if (ptr[1] != 'C' || ptr[2] != 'h') {
    throw std::invalid_argument("Expected 'Ch'");
  }
  uint8_t pm_ch = std::strtol(ptr + 3, &ptr, 10);
  uint8_t ep = side == 'C' ? 1 : 0;
  if (errno) {
    throw std::invalid_argument("Cannot read pm_ch");
  }
  assert(side == 'A' || side == 'C');
  return Topo{pm_num, pm_ch, ep};
}

class LookUpTable
{
  using CcdbManager = o2::ccdb::BasicCCDBManager;
  using CcdbApi = o2::ccdb::CcdbApi;
  static constexpr int NUMBER_OF_MCPs = 12;
  static constexpr int NUMBER_OF_PMs = 19;
  static constexpr int TCM_channel = 228;

 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  // LookUpTable() = default;

  explicit LookUpTable(std::vector<Topo> const& topoVector)
    : mTopoVector(topoVector), mInvTopo(NUMBER_OF_MCPs * 16 * 2)
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo.at(getIdx(mTopoVector[channel].mPM, mTopoVector[channel].mMCP, mTopoVector[channel].mEP)) =
        channel;
    }
  }
  LookUpTable() = default;
  ~LookUpTable() = default;

  int getTCMchannel() const
  {
    return TCM_channel;
  }

  void printFullMap() const
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      std::cout << channel << "\t :  PM \t" << mTopoVector[channel].mPM
                << " MCP \t" << mTopoVector[channel].mMCP << " EP \t " << mTopoVector[channel].mEP << std::endl;
    }
  }

  int getChannel(int link, int mcp, int ep) const
  {
    if ((ep == 0 && (link > 7 && link < 11)) ||
        (ep == 1 && link == 8 && mcp > 8) ||
        (ep == 1 && link == 9 && mcp > 8)) {
      LOG(INFO) << " channel is not conneted "
                << " ep " << ep << " link " << link << " channel " << mcp;
    }
    return mInvTopo[getIdx(link, mcp, ep)];
  }

  int getLink(int channel) const
  {

    return mTopoVector[channel].mPM;
  }
  int getMCP(int channel) const
  {
    return mTopoVector[channel].mMCP;
  }
  int getEP(int channel) const
  {
    return mTopoVector[channel].mEP;
  }

  static o2::ft0::LookUpTable linear()
  {
    std::vector<o2::ft0::Topo> lut_data(NUMBER_OF_MCPs * NUMBER_OF_PMs);
    for (int link = 0; link < NUMBER_OF_PMs; ++link) {
      for (int mcp = 0; mcp < NUMBER_OF_MCPs; ++mcp) {
        lut_data[link * NUMBER_OF_MCPs + mcp] = o2::ft0::Topo{link, mcp};
      }
    }
    return o2::ft0::LookUpTable{lut_data};
  }

  static o2::ft0::LookUpTable readTableFile()
  {
    std::string inputDir;
    const char* aliceO2env = std::getenv("O2_ROOT");
    if (aliceO2env) {
      inputDir = aliceO2env;
    }
    inputDir += "/share/Detectors/FT0/files/";

    std::string lutPath = inputDir + "FT0ChannelsTable.txt";
    lutPath = gSystem->ExpandPathName(lutPath.data()); // Expand $(ALICE_ROOT) into real system path

    std::ifstream infile;
    infile.open(lutPath.c_str());

    std::vector<o2::ft0::Topo> lut_data(NUMBER_OF_MCPs * NUMBER_OF_PMs - 8);
    std::string comment;             // dummy, used just to read 4 first lines and move the cursor to the 5th, otherwise unused
    if (!getline(infile, comment)) { // first comment line
      /* std::cout << "Error opening ascii file (it is probably a folder!): " << filename.c_str() << std::endl; */
      throw std::runtime_error("Error reading lookup table");
    }
    int channel;
    std::string pm, pm_channel, hv_board, hv_channel, mcp_sn, hv_cable, signal_cable;
    std::getline(infile, pm); // skip one line
    std::string line;
    while (std::getline(infile, line) && std::istringstream(line) >> channel >> pm >> pm_channel >> hv_board >> hv_channel >> mcp_sn >> hv_cable >> signal_cable) {
      lut_data[channel] = read_Topo(pm_channel);
    }
    return o2::ft0::LookUpTable{lut_data};
  }
  static o2::ft0::LookUpTable readTable()
  {

    std::vector<o2::ft0::Topo> lut_data;
    auto& mgr = o2::ccdb::BasicCCDBManager::instance();
    mgr.setURL("http://ccdb-test.cern.ch:8080");
    auto hvch = mgr.get<std::vector<o2::ft0::HVchannel>>("FT0/LookUpTable");
    size_t max = 0;
    for (auto const& chan : *hvch) {
      if (max < chan.channel) {
        max = chan.channel;
      }
    }
    lut_data.resize(max + 1);
    for (auto const& chan : *hvch) {
      o2::ft0::Topo topo = chan.pm;
      lut_data[chan.channel] = topo;
    }
    std::cout << "lut_data.size " << lut_data.size() << std::endl;
    return o2::ft0::LookUpTable{lut_data};
  }
  bool isTCM(int link, int ep) const { return getChannel(link, 1, ep) == TCM_channel; }
  Topo getTopoPM(int globalChannelID) const { return mTopoVector[globalChannelID]; }
  Topo getTopoTCM() const { return mTopoVector[TCM_channel]; }
  std::size_t getNchannels() const { return TCM_channel; } //get number of global PM channels
 private:
  std::vector<Topo> mTopoVector;
  std::vector<int> mInvTopo;

  static int getIdx(int link, int mcp, int ep)
  {
    assert(mcp < NUMBER_OF_MCPs);
    /* if ((ep == 0 && (link > 7 && link < 11)) || */
    /*     (ep == 1 && link == 8 && mcp > 8) || */
    /*     (ep == 1 && link == 9 && mcp > 8)) { */
    /*   LOG(INFO)<<" channel is not conneted "<<" ep "<<ep<<" link "<<link<<" channel "<<mcp; */
    /*   return 255; */
    /* } */
    return (link + ep * 16) * NUMBER_OF_MCPs + mcp;
  }
  static int getLinkFromIdx(int idx)
  {
    int link;
    if (idx > 95) {
      link = (idx - 96) / NUMBER_OF_MCPs;
    } else {
      link = idx / NUMBER_OF_MCPs;
    }
    return link;
  }
  static int getEPFromIdx(int idx)
  {
    if (idx < 96 || idx > 215) {
      return 0;
    } else {
      return 1;
    }
  }

  static int getMCPFromIdx(int idx) { return idx % NUMBER_OF_MCPs + 1; }

  ClassDefNV(LookUpTable, 2);
};

//Singleton for LookUpTable
class SingleLUT : public LookUpTable
{
 private:
  SingleLUT() : LookUpTable(LookUpTable::readTable()) {}
  SingleLUT(const SingleLUT&) = delete;
  SingleLUT& operator=(SingleLUT&) = delete;

 public:
  typedef Topo Topo_t;
  static constexpr char sDetectorName[] = "FT0";
  static SingleLUT& Instance()
  {
    static SingleLUT instanceLUT;
    return instanceLUT;
  }
  //Temporary
  //Making topo for FEE recognizing(Local channelID is supressed)
  static Topo_t makeGlobalTopo(const Topo_t& topo)
  {
    return Topo_t{topo.mPM, 0, topo.mEP};
  }
  static int getLocalChannelID(const Topo_t& topo)
  {
    return topo.mMCP;
  }
  //Prepare full map for FEE metadata
  template <typename RDHtype, typename RDHhelper = void>
  auto makeMapFEEmetadata() -> std::map<Topo_t, RDHtype>
  {
    std::map<Topo_t, RDHtype> mapResult;
    const uint16_t cruID = 0;      //constant
    const uint32_t endPointID = 0; //constant
    uint64_t feeID = 0;            //increments
    //PM
    for (int iCh = 0; iCh < Instance().getNchannels(); iCh++) {
      auto pairInserted = mapResult.insert({makeGlobalTopo(Instance().getTopoPM(iCh)), RDHtype{}});
      if (pairInserted.second) {
        auto& rdhObj = pairInserted.first->second;
        const auto& topoObj = pairInserted.first->first;
        if constexpr (std::is_same<RDHhelper, void>::value) {
          rdhObj.linkID = topoObj.mPM;
          rdhObj.endPointID = topoObj.mEP;
          rdhObj.feeId = feeID;
          rdhObj.cruID = cruID;
        } else //Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.mPM);
          RDHhelper::setEndPointID(&rdhObj, topoObj.mEP);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
        feeID++;
      }
    }
    //TCM
    {
      auto pairInserted = mapResult.insert({makeGlobalTopo(Instance().getTopoTCM()), RDHtype{}});
      if (pairInserted.second) {
        auto& rdhObj = pairInserted.first->second;
        const auto& topoObj = pairInserted.first->first;
        if constexpr (std::is_same<RDHhelper, void>::value) {
          rdhObj.linkID = topoObj.mPM;
          rdhObj.endPointID = topoObj.mEP;
          rdhObj.feeId = feeID;
          rdhObj.cruID = cruID;
        } else //Using RDHUtils
        {
          RDHhelper::setLinkID(&rdhObj, topoObj.mPM);
          RDHhelper::setEndPointID(&rdhObj, topoObj.mEP);
          RDHhelper::setFEEID(&rdhObj, feeID);
          RDHhelper::setCRUID(&rdhObj, cruID);
        }
      } else {
        LOG(INFO) << "WARNING! CHECK LUT! TCM METADATA IS INCORRECT!";
      }
    }
    assert(mapResult.size() > 0);
    return mapResult;
  }
};
} // namespace ft0
} // namespace o2
#endif
