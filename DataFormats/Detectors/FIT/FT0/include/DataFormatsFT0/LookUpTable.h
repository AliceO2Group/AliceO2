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

namespace o2
{
namespace ft0
{
struct Topo {
  int mPM = 0;
  int mMCP = 0;
  ClassDefNV(Topo, 1);
};

// enum class Side : char { A, C };
// struct PM {
//   Side side;
//   uint8_t PM_num, PM_channel;
// };

struct HVchannel {
  enum class HVBoard : uint8_t { NA,
                                 A_out,
                                 A_in,
                                 C_up,
                                 C_down,
                                 C_mid };
  uint8_t channel;
  Topo pm;
  HVBoard HV_board;
  uint8_t HV_channel;
  std::string MCP_SN;
  std::string HV_cabel;
  std::string signal_cable;

  ClassDefNV(HVchannel, 1);
};

inline bool operator<(Topo const& a, Topo const& b)
{
  return (a.mPM < b.mPM || (a.mPM == b.mPM && a.mMCP < b.mMCP));
}

inline o2::ft0::Topo read_Topo(std::string_view str)
{
  assert(str.substr(0, 2) == "PM" && str[4] == '/' && str[5] == 'C' && str[6] == 'h');
  char side = str[2];
  uint8_t pm_num = str[3] - '0';
  uint8_t pm_ch = (str[7] - '0') * 10 + (str[8] - '0') - 1;
  assert(side == 'A' || side == 'C');
  if (str.substr(0, 4) == "PMA9") {
    pm_num = 18;
  }
  return Topo{(side == 'C' ? 8 : 0) + pm_num, pm_ch};
}

class LookUpTable
{
  using CcdbManager = o2::ccdb::BasicCCDBManager;
  using CcdbApi = o2::ccdb::CcdbApi;
  static constexpr int NUMBER_OF_MCPs = 12;
  static constexpr int NUMBER_OF_PMs = 19;

 public:
  ///
  /// Default constructor.
  /// It must be kept public for root persistency purposes,
  /// but should never be called by the outside world
  // LookUpTable() = default;

  explicit LookUpTable(std::vector<Topo> const& topoVector)
    : mTopoVector(topoVector), mInvTopo(topoVector.size())
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      mInvTopo.at(getIdx(mTopoVector[channel].mPM, mTopoVector[channel].mMCP)) =
        channel;
    }
  }
  LookUpTable() = default;
  ~LookUpTable() = default;

  HVchannel mHVchannel;

  void printFullMap() const
  {
    for (size_t channel = 0; channel < mTopoVector.size(); ++channel) {
      std::cout << channel << "\t :  PM \t" << mTopoVector[channel].mPM
                << " MCP \t" << mTopoVector[channel].mMCP << std::endl;
    }
    for (size_t idx = 0; idx < mInvTopo.size(); ++idx) {
      std::cout << "PM \t" << getLinkFromIdx(mInvTopo[idx]) << " MCP \t"
                << getMCPFromIdx(mInvTopo[idx]) << std::endl;
    }
  }

  int getChannel(int link, int mcp) const
  {
    return mInvTopo[getIdx(link, mcp)];
  }

  int getLink(int channel) const
  {
    return mTopoVector[channel].mPM;
  }
  int getMCP(int channel) const
  {
    return mTopoVector[channel].mMCP;
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
    while (infile >> channel >> pm >> pm_channel >> hv_board >> hv_channel >> mcp_sn >> hv_cable >> signal_cable) {
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
    return o2::ft0::LookUpTable{lut_data};
  }

 private:
  std::vector<Topo> mTopoVector;
  std::vector<int> mInvTopo;

  static int getIdx(int link, int mcp)
  {
    assert(mcp < NUMBER_OF_MCPs);
    return link * NUMBER_OF_MCPs + mcp;
  }
  static int getLinkFromIdx(int idx) { return idx / NUMBER_OF_MCPs; }
  static int getMCPFromIdx(int idx) { return idx % NUMBER_OF_MCPs; }

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
  static SingleLUT& Instance()
  {
    static SingleLUT instanceLUT;
    return instanceLUT;
  }
};
} // namespace ft0
} // namespace o2
#endif
