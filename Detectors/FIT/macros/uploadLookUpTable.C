#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include <DataFormatsFT0/LookUpTable.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <string_view>

o2::ft0::HVchannel::HVBoard readHVBoard(std::string_view str);
o2::ft0::Topo read_Topo(std::string_view str);

void uploadLookUpTable()
{
  using o2::ccdb::BasicCCDBManager;
  using o2::ccdb::CcdbApi;

  //  std::ifstream f{filename.data()};
  std::ifstream f{"FT0ChannelsTable.txt"};
  int channel;
  std::string pm, pm_channel, hv_board, hv_channel, mcp_sn, hv_cable, signal_cable;
  std::vector<o2::ft0::HVchannel> table;
  std::getline(f, pm); // skip one line
  while (f >> channel >> pm >> pm_channel >> hv_board >> hv_channel >> mcp_sn >> hv_cable >> signal_cable) {
    o2::ft0::HVchannel chan;
    chan.channel = channel;
    assert(std::string_view(pm_channel).substr(2, 2) == pm);
    chan.pm = read_Topo(pm_channel);
    chan.HV_board = readHVBoard(hv_board);
    chan.MCP_SN = mcp_sn;
    chan.HV_cabel = hv_cable;
    chan.signal_cable = signal_cable;
    table.emplace_back(chan);
  }
  CcdbApi api;
  std::map<std::string, std::string> metadata; // can be empty
  api.init("http://ccdb-test.cern.ch:8080/");  // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&table, "FT0/LookUpTable", metadata);
}

o2::ft0::HVchannel::HVBoard readHVBoard(std::string_view str)
{
  using HVBoard = o2::ft0::HVchannel::HVBoard;
  if (str == "N/A")
    return HVBoard::NA;
  else if (str == "A-Out")
    return HVBoard::A_out;
  else if (str == "A-In")
    return HVBoard::A_in;
  else if (str == "C-Up")
    return HVBoard::C_up;
  else if (str == "C-Down")
    return HVBoard::C_down;
  else if (str == "C-Mid")
    return HVBoard::C_mid;
  else {
    std::cerr << "Unknown HVBoard " << str << "\n";
    std::abort();
  }
}
o2::ft0::Topo read_Topo(std::string_view str)
{
  assert(str.substr(0, 2) == "PM" && str[4] == '/' && str[5] == 'C' && str[6] == 'h');
  char side = str[2];
  uint8_t pm_num = str[3] - '0';
  uint8_t pm_ch = (str[7] - '0') * 10 + (str[8] - '0');
  assert(side == 'A' || side == 'C');
  return {(side == 'C' ? 10 : 0) + pm_num, pm_ch};
}
