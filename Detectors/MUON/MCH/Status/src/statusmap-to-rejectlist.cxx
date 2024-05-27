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

/// @author Philippe Pillot

#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <boost/program_options.hpp>

#include <TFile.h>
#include <TTreeReader.h>
#include <TTreeReaderValue.h>

#include "CCDB/CcdbApi.h"
#include "DataFormatsMCH/DsChannelId.h"
#include "Framework/Logger.h"
#include "MCHGlobalMapping/ChannelCode.h"
#include "MCHStatus/StatusMap.h"

namespace po = boost::program_options;

using BadChannelsVector = std::vector<o2::mch::DsChannelId>;

//____________________________________________________________________________________
std::tuple<TFile*, TTreeReader*> loadData(const std::string inFile)
{
  /// open the input file and get the intput tree

  TFile* f = TFile::Open(inFile.c_str(), "READ");
  if (!f || f->IsZombie()) {
    LOG(error) << "opening file " << inFile << " failed";
    exit(2);
  }

  TTreeReader* r = new TTreeReader("o2sim", f);
  if (r->IsZombie()) {
    LOG(error) << "tree o2sim not found";
    exit(2);
  }

  return std::make_tuple(f, r);
}

//____________________________________________________________________________________
int size(const std::map<int, std::vector<int>>& badChannels)
{
  /// return the total number of bad channels

  int n = 0;

  for (const auto& channels : badChannels) {
    n += channels.second.size();
  }

  return n;
};

//____________________________________________________________________________________
void printContent(const std::string inFile, const uint32_t mask)
{
  /// print the content of the status maps with the given mask

  auto [dataFile, dataReader] = loadData(inFile);
  TTreeReaderValue<o2::mch::StatusMap> statusMap(*dataReader, "statusmaps");

  int iTF(-1);
  int firstTF(-1);
  int lastTF(-1);
  std::map<int, std::vector<int>> currentBadChannels{};

  while (dataReader->Next()) {
    ++iTF;

    // record the first status map
    if (firstTF < 0) {
      firstTF = iTF;
      lastTF = iTF;
      currentBadChannels = o2::mch::applyMask(*statusMap, mask);
      continue;
    }

    auto badChannels = o2::mch::applyMask(*statusMap, mask);

    // extend the TF range of the current status map if it did not change
    if (badChannels == currentBadChannels) {
      lastTF = iTF;
      continue;
    }

    // print the current status map
    LOGP(info, "TF [{}, {}]: the status map contains {} bad channels in {} detection element{} (using statusMask=0x{:x})",
         firstTF, lastTF, size(currentBadChannels), currentBadChannels.size(), currentBadChannels.size() > 1 ? "s" : "", mask);

    // update the current status map
    firstTF = iTF;
    lastTF = iTF;
    currentBadChannels = badChannels;
  }

  // print the last status map
  LOGP(info, "TF [{}, {}]: the status map contains {} bad channels in {} detection element{} (using statusMask=0x{:x})",
       firstTF, lastTF, size(currentBadChannels), currentBadChannels.size(), currentBadChannels.size() > 1 ? "s" : "", mask);

  dataFile->Close();
}

//____________________________________________________________________________________
BadChannelsVector statusMap2RejectList(const std::string inFile, const size_t iTF, const uint32_t mask)
{
  /// convert the status map of the given TF into a reject list with the given mask

  auto [dataFile, dataReader] = loadData(inFile);
  TTreeReaderValue<o2::mch::StatusMap> statusMap(*dataReader, "statusmaps");

  if (dataReader->SetEntry(iTF) != TTreeReader::kEntryValid) {
    LOGP(error, "invalid TF index {} (number of TFs = {})", iTF, dataReader->GetEntries());
    exit(3);
  }

  BadChannelsVector bv;

  for (const auto& status : *statusMap) {
    auto channel = status.first;
    if (!channel.isValid()) {
      LOG(error) << "invalid channel";
    }
    if ((mask & status.second) != 0) {
      const auto c = o2::mch::DsChannelId(channel.getSolarId(), channel.getElinkId(), channel.getChannel());
      bv.emplace_back(c);
    }
  }

  dataFile->Close();

  LOGP(info, "the reject list contains {} bad channels (using statusMask=0x{:x})", bv.size(), mask);

  return bv;
}

//____________________________________________________________________________________
void uploadRejectList(const std::string ccdbUrl, uint64_t startTS, uint64_t endTS, const BadChannelsVector& bv)
{
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;

  LOGP(info, "storing MCH RejectList (valid from {} to {}) to MCH/Calib/RejectList",
       startTS, endTS);

  api.storeAsTFileAny(&bv, "MCH/Calib/RejectList", md, startTS, endTS);
}

//____________________________________________________________________________________
int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  uint64_t startTS;
  uint64_t endTS;
  std::string inFile;
  size_t iTF;
  uint32_t mask;
  bool print;

  auto tnow = std::chrono::system_clock::now().time_since_epoch();
  using namespace std::chrono_literals;
  auto tend = tnow + 24h;
  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(tnow).count();
  uint64_t end = std::chrono::duration_cast<std::chrono::milliseconds>(tend).count();

  uint32_t defaultMask = o2::mch::StatusMap::kBadPedestal | o2::mch::StatusMap::kRejectList | o2::mch::StatusMap::kBadHV;

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("starttimestamp,st",po::value<uint64_t>(&startTS)->default_value(now),"timestamp for query or put - (default=now)")
      ("endtimestamp,et", po::value<uint64_t>(&endTS)->default_value(end), "end of validity (for put) - default=1 day from now")
      ("infile,f",po::value<std::string>(&inFile)->default_value("mchstatusmaps.root"),"input file of StatusMap objects")
      ("tf,i", po::value<size_t>(&iTF)->default_value(0), "index of the TF to process")
      ("mask,m", po::value<uint32_t>(&mask)->default_value(defaultMask), "mask to apply to the statusMap to produce the RejectList")
      ("print,p",po::bool_switch(&print),"print the content of the input file without processing it")
        ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    LOG(info) << "This program converts a StatusMap to a RejectList CCDB object";
    LOG(info) << usage;
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    LOG(error) << e.what();
    exit(1);
  }

  if (print) {
    printContent(inFile, mask);
  } else {
    auto bv = statusMap2RejectList(inFile, iTF, mask);
    uploadRejectList(ccdbUrl, startTS, endTS, bv);
  }

  return 0;
}
