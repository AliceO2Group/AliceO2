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

#include <boost/program_options.hpp>
#include <ctime>
#include <fstream>
#include <iterator>
#include <numeric>
#include <regex>
#include <set>
#include <string>
#include <vector>
#include "CCDB/CcdbApi.h"
#include "DataFormatsMCH/DsChannelId.h"
#include "MCHConditions/DCSAliases.h"
#include "MCHConstants/DetectionElements.h"
#include "MCHGlobalMapping/DsIndex.h"
#include "MCHGlobalMapping/Mapper.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHRawElecMap/Mapper.h"

namespace po = boost::program_options;

using BadChannelsVector = std::vector<o2::mch::DsChannelId>;

std::string ccdbPath(const std::string badChannelType)
{
  return fmt::format("MCH/Calib/{}", badChannelType);
}

std::set<uint64_t> listTSWhenBadChannelsChange(const std::string ccdbUrl, const std::string badChannelType,
                                               uint64_t startTimestamp, uint64_t endTimestamp, bool verbose)
{
  std::set<uint64_t> tsChanges{startTimestamp};

  std::cout << std::endl;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  auto source = ccdbPath(badChannelType);

  // store every timestamps in the time range when the bad channels potentially change
  if (verbose) {
    std::cout << fmt::format("\nlist of {} files potentially valid in the range {} - {}:\n", source, startTimestamp, endTimestamp);
  }
  std::istringstream fileInfo(api.list(source, false, "text/plain"));
  std::string dummy{};
  std::string path{};
  uint64_t begin = 0;
  uint64_t end = 0;
  uint64_t creation = 0;
  bool inRange = false;
  for (std::string line; std::getline(fileInfo, line);) {
    if (line.find("Validity:") == 0) {
      std::istringstream in(line);
      in >> dummy >> begin >> dummy >> end;
      if (begin < endTimestamp && end > startTimestamp) {
        if (begin >= startTimestamp) {
          tsChanges.emplace(begin);
        }
        if (end < endTimestamp) {
          tsChanges.emplace(end);
        }
        inRange = true;
      }
    } else if (verbose) {
      if (line.find("ID:") == 0) {
        std::istringstream in(line);
        in >> dummy >> path;
      } else if (inRange && line.find("Created:") == 0) {
        std::istringstream in(line);
        in >> dummy >> creation;
        std::cout << fmt::format("- {}\n", path);
        std::cout << fmt::format("  validity range: {} - {}\n", begin, end);
        std::cout << fmt::format("  creation time: {}\n", creation);
        inRange = false;
      }
    }
  }
  if (verbose) {
    std::cout << fmt::format("\nlist of timestamps when the bad channels potentially change:\n");
    for (auto ts : tsChanges) {
      std::cout << fmt::format("  {}\n", ts);
    }
  }

  // select timestamps when the bad channels actually change
  if (verbose) {
    std::cout << fmt::format("\nlist of {} files actually valid in the range {} - {}:\n", source, startTimestamp, endTimestamp);
  }
  std::map<std::string, std::string> metadata{};
  std::string currentETag{};
  for (auto itTS = tsChanges.begin(); itTS != tsChanges.end();) {
    auto headers = api.retrieveHeaders(source, metadata, *itTS);
    if (headers["ETag"] == currentETag) {
      itTS = tsChanges.erase(itTS);
    } else {
      if (verbose) {
        std::cout << fmt::format("- {}\n", headers["Location"]);
        std::cout << fmt::format("  validity range: {} - {}\n", headers["Valid-From"], headers["Valid-Until"]);
        std::cout << fmt::format("  creation time: {}\n", headers["Created"]);
      }
      currentETag = headers["ETag"];
      ++itTS;
    }
  }
  std::cout << fmt::format("\nlist of timestamps when the bad channels actually change:\n");
  for (auto ts : tsChanges) {
    std::cout << fmt::format("  {}\n", ts);
  }

  return tsChanges;
}

BadChannelsVector queryBadChannels(const std::string ccdbUrl,
                                   const std::string badChannelType, uint64_t timestamp, bool verbose)
{
  std::cout << std::endl;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> metadata;
  auto source = ccdbPath(badChannelType);
  auto* badChannels = api.retrieveFromTFileAny<BadChannelsVector>(source, metadata, timestamp);
  std::cout << "number of bad channels = " << badChannels->size() << std::endl;
  if (verbose) {
    for (const auto& badChannel : *badChannels) {
      std::cout << badChannel.asString() << "\n";
    }
  }
  return *badChannels;
}

void rejectDS(const o2::mch::raw::DsDetId& dsDetId, BadChannelsVector& bv)
{
  static auto det2elec = o2::mch::raw::createDet2ElecMapper<o2::mch::raw::ElectronicMapperGenerated>();
  const auto& seg = o2::mch::mapping::segmentation(dsDetId.deId());
  auto dsElecId = det2elec(dsDetId);
  seg.forEachPadInDualSampa(dsDetId.dsId(), [&](int pad) {
    uint8_t channel = seg.padDualSampaChannel(pad);
    const auto c = o2::mch::DsChannelId(dsElecId->solarId(), dsElecId->elinkId(), channel);
    bv.emplace_back(c);
  });
}

void rejectSolars(const std::vector<uint16_t> solarIds, BadChannelsVector& bv)
{
  for (auto solar : solarIds) {
    auto ds = o2::mch::raw::getDualSampas<o2::mch::raw::ElectronicMapperGenerated>(solar);
    for (const auto& dsDetId : ds) {
      rejectDS(dsDetId, bv);
    }
  }
}

void rejectDSs(const std::vector<uint16_t> dsIdxs, BadChannelsVector& bv)
{
  for (auto ds : dsIdxs) {
    if (ds >= o2::mch::NumberOfDualSampas) {
      std::cout << "Error: invalid DS index " << ds << std::endl;
      continue;
    }
    o2::mch::raw::DsDetId dsDetId = o2::mch::getDsDetId(ds);
    rejectDS(dsDetId, bv);
  }
}

void rejectHVLVs(const std::vector<std::string> dcsAliases, BadChannelsVector& bv)
{
  for (auto alias : dcsAliases) {
    if (!o2::mch::dcs::isValid(alias)) {
      std::cout << "Error: invalid alias " << alias << std::endl;
      continue;
    }
    for (auto ds : o2::mch::dcs::aliasToDsIndices(alias)) {
      o2::mch::raw::DsDetId dsDetId = o2::mch::getDsDetId(ds);
      rejectDS(dsDetId, bv);
    }
  }
}

void rejectDEs(const std::vector<uint16_t> deIds, BadChannelsVector& bv)
{
  static auto det2elec = o2::mch::raw::createDet2ElecMapper<o2::mch::raw::ElectronicMapperGenerated>();
  for (auto de : deIds) {
    if (!o2::mch::constants::isValidDetElemId(de)) {
      std::cout << "Error: invalid DE ID " << de << std::endl;
      continue;
    }
    const auto& seg = o2::mch::mapping::segmentation(de);
    seg.forEachPad([&](int pad) {
      auto ds = seg.padDualSampaId(pad);
      o2::mch::raw::DsDetId dsDetId(de, ds);
      auto dsElecId = det2elec(dsDetId);
      uint8_t channel = seg.padDualSampaChannel(pad);
      const auto c = o2::mch::DsChannelId(dsElecId->solarId(), dsElecId->elinkId(), channel);
      bv.emplace_back(c);
    });
  }
}

void uploadBadChannels(const std::string ccdbUrl,
                       const std::string badChannelType,
                       uint64_t startTimestamp,
                       uint64_t endTimestamp,
                       const BadChannelsVector& bv,
                       bool makeDefault)
{
  std::cout << std::endl;
  o2::ccdb::CcdbApi api;
  api.init(ccdbUrl);
  std::map<std::string, std::string> md;
  auto dest = ccdbPath(badChannelType);
  std::cout << fmt::format("storing {} {}bad channels (valid from {} to {}) to {}\n", bv.size(),
                           makeDefault ? "default " : "", startTimestamp, endTimestamp, dest);

  if (makeDefault) {
    md["default"] = "true";
    md["Created"] = "1";
  }
  api.storeAsTFileAny(&bv, dest, md, startTimestamp, endTimestamp);
}

int main(int argc, char** argv)
{
  po::variables_map vm;
  po::options_description usage("Usage");

  std::string ccdbUrl;
  std::string ccdbRefUrl;
  std::string dpConfName;
  std::string badChannelType;
  uint64_t startTimestamp;
  uint64_t endTimestamp;
  bool list;
  bool put;
  bool query;
  bool verbose;
  bool uploadDefault;

  auto tnow = std::chrono::system_clock::now().time_since_epoch();
  using namespace std::chrono_literals;
  auto tend = tnow + 24h;

  uint64_t now = std::chrono::duration_cast<std::chrono::milliseconds>(tnow).count();
  uint64_t end = std::chrono::duration_cast<std::chrono::milliseconds>(tend).count();

  // clang-format off
  usage.add_options()
      ("help,h", "produce help message")
      ("ccdb,c",po::value<std::string>(&ccdbUrl)->default_value("http://localhost:6464"),"ccdb url")
      ("starttimestamp",po::value<uint64_t>(&startTimestamp)->default_value(now),"timestamp for query or put - (default=now)")
      ("endtimestamp", po::value<uint64_t>(&endTimestamp)->default_value(end), "end of validity (for put) - default=1 day from now")
      ("list,l", po::bool_switch(&list),"list timestamps, within the given range, when the bad channels change")
      ("put,p",po::bool_switch(&put),"upload bad channel object")
      ("referenceccdb,r",po::value<std::string>(&ccdbRefUrl)->default_value("http://alice-ccdb.cern.ch"),"reference ccdb url")
      ("upload-default-values,u",po::bool_switch(&uploadDefault),"upload default values")
      ("type,t",po::value<std::string>(&badChannelType)->default_value("BadChannel"),"type of bad channel (BadChannel or RejectList)")
      ("query,q",po::bool_switch(&query),"dump bad channel object from CCDB")
      ("verbose,v",po::bool_switch(&verbose),"verbose output")
      ("solar,s",po::value<std::vector<uint16_t>>()->multitoken(),"solar ids to reject")
      ("ds,d", po::value<std::vector<uint16_t>>()->multitoken(), "dual sampas indices to reject")
      ("de,e", po::value<std::vector<uint16_t>>()->multitoken(), "DE ids to reject")
      ("alias,a", po::value<std::vector<std::string>>()->multitoken(), "DCS alias (HV or LV) to reject")
        ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(usage);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << "This program get/set MCH bad channels CCDB object\n";
    std::cout << usage << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  if (badChannelType != "BadChannel" && badChannelType != "RejectList") {
    std::cout << "Error: badChannelType " << badChannelType << " is invalid. Only BadChannel or RejectList are legit\n";
    exit(2);
  }

  if (list) {
    auto tsChanges = listTSWhenBadChannelsChange(ccdbUrl, badChannelType, startTimestamp, endTimestamp, verbose);
  }

  if (query) {
    queryBadChannels(ccdbUrl, badChannelType, startTimestamp, verbose);
  }

  if (put) {
    BadChannelsVector bv;
    if (vm.count("solar")) {
      rejectSolars(vm["solar"].as<std::vector<uint16_t>>(), bv);
    }
    if (vm.count("ds")) {
      rejectDSs(vm["ds"].as<std::vector<uint16_t>>(), bv);
    }
    if (vm.count("de")) {
      rejectDEs(vm["de"].as<std::vector<uint16_t>>(), bv);
    }
    if (vm.count("alias")) {
      rejectHVLVs(vm["alias"].as<std::vector<std::string>>(), bv);
    }

    if (uploadDefault) {
      uploadBadChannels(ccdbUrl, badChannelType, startTimestamp, endTimestamp, bv, true);
    } else {
      auto tsChanges = listTSWhenBadChannelsChange(ccdbRefUrl, badChannelType, startTimestamp, endTimestamp, false);

      std::cout << fmt::format("\n{} object{} valid in the reference CCDB ({}) for this time range. What do you want to do?\n",
                               tsChanges.size(), (tsChanges.size() > 1) ? "s are" : " is", ccdbRefUrl);
      std::cout << fmt::format("[a] abort: do nothing\n");
      std::cout << fmt::format("[o] overwrite: create 1 new object for the whole time range to supersede the existing one{}\n",
                               (tsChanges.size() > 1) ? "s" : "");
      std::cout << fmt::format("[u] update: create {} new object{} within the time range adding new bad channels to existing ones\n",
                               tsChanges.size(), (tsChanges.size() > 1) ? "s" : "");
      std::string response{};
      std::cin >> response;

      if (response == "a" || response == "abort") {
        return 0;
      } else if (response == "o" || response == "overwrite") {
        uploadBadChannels(ccdbUrl, badChannelType, startTimestamp, endTimestamp, bv, false);
      } else if (response == "u" || response == "update") {
        tsChanges.emplace(endTimestamp);
        auto itStartTS = tsChanges.begin();
        for (auto itStopTS = std::next(itStartTS); itStopTS != tsChanges.end(); ++itStartTS, ++itStopTS) {
          auto bv2 = queryBadChannels(ccdbRefUrl, badChannelType, *itStartTS, false);
          bv2.insert(bv2.end(), bv.begin(), bv.end());
          uploadBadChannels(ccdbUrl, badChannelType, *itStartTS, *itStopTS, bv2, false);
        }
      } else {
        std::cout << "invalid response (must be a, o or u) --> abort\n";
        exit(3);
      }
    }
  }

  return 0;
}
