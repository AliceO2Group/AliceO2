#include "CCDB/BasicCCDBManager.h"
#include "CCDB/CcdbApi.h"
#include <DataFormatsFT0/LookUpTable.h>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <iostream>
#include <map>
#include <string_view>

//o2::ft0::HVchannel::HVBoard readHVBoard(std::string_view str);

void uploadLookUpTable()
{
  using o2::ccdb::BasicCCDBManager;
  using o2::ccdb::CcdbApi;
  using namespace o2::ft0;

  std::ifstream f{"/home/alla/aliO2/O2/Detectors/FIT/FT0/base/files/FT0ChannelsTable.02.0.4.2021.txt"};
  int channel;
  std::string pm, pm_channel, hv_board, hv_channel, mcp_sn, hv_cable, signal_cable, ep_PM;
  std::vector<o2::ft0::HVchannel> table;
  std::getline(f, pm); // skip one line
  std::string line;
  while (std::getline(f, line) && (std::istringstream(line) >> channel >> pm >> pm_channel >> hv_board >> hv_channel >> mcp_sn >> hv_cable >> signal_cable >> ep_PM)) {
    o2::ft0::HVchannel chan;
    chan.channel = channel;
    assert(std::string_view(pm_channel).substr(2, 2) == pm);
    chan.pm = read_Topo(pm_channel);
    chan.HV_board = hv_board; //readHVBoard(hv_board);
    chan.MCP_SN = mcp_sn;
    chan.HV_cabel = hv_cable;
    chan.signal_cable = signal_cable;
    chan.EP = ep_PM;
    LOG(INFO) << "read channel " << int(chan.channel);
    table.emplace_back(chan);
  }
  CcdbApi api;
  std::map<std::string, std::string> metadata; // can be empty
  api.init("http://ccdb-test.cern.ch:8080/");  // or http://localhost:8080 for a local installation
  // store abitrary user object in strongly typed manner
  api.storeAsTFileAny(&table, "FT0/LookUpTable", metadata);
}
/*
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
*/
