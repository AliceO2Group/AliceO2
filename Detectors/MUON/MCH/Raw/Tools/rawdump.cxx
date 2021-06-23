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

///
/// @author  Laurent Aphecetche

#include "DetectorsRaw/RDHUtils.h"
#include "DumpBuffer.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/PageDecoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "boost/program_options.hpp"
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <gsl/span>
#include <iostream>
#include <map>
#include <optional>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <string>

namespace po = boost::program_options;

namespace o2::header
{
extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);
}

using namespace o2::mch::raw;
using RDHv4 = o2::header::RAWDataHeaderV4;

class DumpOptions
{
 public:
  DumpOptions(unsigned int maxNofRDHs, bool showRDHs, bool jsonOutput)
    : mMaxNofRDHs{maxNofRDHs == 0 ? std::numeric_limits<unsigned int>::max() : maxNofRDHs}, mShowRDHs{showRDHs}, mJSON{jsonOutput} {}

  unsigned int maxNofRDHs() const
  {
    return mMaxNofRDHs;
  }

  bool showRDHs() const
  {
    return mShowRDHs;
  }

  bool json() const
  {
    return mJSON;
  }

  std::optional<uint16_t> cruId() const
  {
    return mCruId;
  }

  void cruId(uint16_t c) { mCruId = c; }

 private:
  unsigned int mMaxNofRDHs;
  bool mShowRDHs;
  bool mJSON;
  std::optional<uint16_t> mCruId{std::nullopt};
};

struct ChannelStat {
  double mean{0};
  double rms{0};
  double q{0};
  int n{0};

  void incr(int v)
  {
    n++;
    auto newMean = mean + (v - mean) / n;
    q += (v - newMean) * (v - mean);
    rms = sqrt(q / n);
    mean = newMean;
  }
};

std::ostream& operator<<(std::ostream& os, const ChannelStat& s)
{
  os << fmt::format("MEAN {:7.3f} RMS {:7.3f} NSAMPLES {:5d} ", s.mean, s.rms, s.n);
  return os;
}

std::map<std::string, ChannelStat> rawdump(std::string input, DumpOptions opt)
{
  std::ifstream in(input.c_str(), std::ios::binary);
  if (!in.good()) {
    std::cout << "could not open file " << input << "\n";
    return {};
  }
  constexpr size_t pageSize = 8192;

  std::array<std::byte, pageSize> buffer;
  gsl::span<const std::byte> page(buffer);

  size_t nrdhs{0};

  const auto patchPage = [&](gsl::span<std::byte> rdhBuffer) {
    auto rdhPtr = const_cast<void*>(reinterpret_cast<const void*>(rdhBuffer.data()));
    nrdhs++;
    auto cruId = o2::raw::RDHUtils::getCRUID(rdhPtr);
    if (opt.cruId().has_value()) {
      cruId = opt.cruId().value();
      o2::raw::RDHUtils::setLinkID(rdhPtr, 0);
    }
    auto endpoint = o2::raw::RDHUtils::getEndPointID(rdhPtr);
    o2::raw::RDHUtils::setFEEID(rdhPtr, cruId * 2 + endpoint);
    if (opt.showRDHs()) {
      std::cout << nrdhs << "--\n";
      o2::raw::RDHUtils::printRDH(rdhPtr);
    }
  };

  struct Counters {
    std::map<std::string, int> uniqueDS;
    std::map<std::string, int> uniqueChannel;
    std::map<std::string, ChannelStat> statChannel;
    int ndigits{0};
  } counters;

  const auto channelHandler =
    [&](DsElecId dsId, uint8_t channel, o2::mch::raw::SampaCluster sc) {
      auto s = asString(dsId);
      counters.uniqueDS[s]++;
      auto ch = fmt::format("{}-CH{}", s, channel);
      counters.uniqueChannel[ch]++;
      auto& chanstat = counters.statChannel[ch];
      if (sc.isClusterSum()) {
        chanstat.incr(sc.chargeSum);
      } else {
        for (auto d = 0; d < sc.nofSamples(); d++) {
          chanstat.incr(sc.samples[d]);
        }
      }
      counters.ndigits++;
    };

  o2::mch::raw::PageDecoder decode = nullptr;

  size_t npages{0};
  uint64_t bytesRead{0};

  char* cbuffer = reinterpret_cast<char*>(&buffer[0]);
  char* cpayload = reinterpret_cast<char*>(&buffer[sizeof(o2::header::RDHAny)]);
  const void* rdhPtr = reinterpret_cast<const void*>(&buffer[0]);

  while (npages < opt.maxNofRDHs()) {

    // read the next RDH, stop if no more data is available
    in.read(cbuffer, sizeof(o2::header::RDHAny));
    if (in.fail()) {
      break;
    }

    auto readSize = o2::raw::RDHUtils::getOffsetToNext(rdhPtr) - sizeof(o2::header::RDHAny);
    in.read(cpayload, readSize);
    if (in.fail()) {
      break;
    }

    npages++;
    bytesRead += in.gcount();
    if (!decode) {
      DecodedDataHandlers handlers;
      handlers.sampaChannelHandler = channelHandler;
      decode = createPageDecoder(page, handlers);
    }
    patchPage(buffer);
    decode(page);
  }

  if (!opt.json()) {
    std::cout << counters.ndigits << " digits seen - " << nrdhs << " RDHs seen - " << npages << " npages read\n";
    std::cout << "#unique DS=" << counters.uniqueDS.size() << " #unique Channel=" << counters.uniqueChannel.size() << "\n";
  }

  return counters.statChannel;
}

void output(const std::map<std::string, ChannelStat>& channels)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  writer.StartObject();
  writer.Key("channels");
  writer.StartArray();
  for (auto s : channels) {
    writer.StartObject();
    writer.Key("id");
    writer.String(s.first.c_str());
    writer.Key("ped");
    writer.Double(s.second.mean);
    writer.Key("noise");
    writer.Double(s.second.rms);
    writer.Key("nof_samples");
    writer.Int(s.second.n);
    writer.EndObject();
  }
  writer.EndArray();
  writer.EndObject();
  std::cout << buffer.GetString() << "\n";
}

int main(int argc, char* argv[])
{
  std::string prefix;
  std::vector<int> detElemIds;
  std::string inputFile;
  po::variables_map vm;
  po::options_description generic("Generic options");
  unsigned int nrdhs{0};
  bool showRDHs{false};
  bool userLogic{false}; // default format is bareformat...
  bool chargeSum{false}; //... in sample mode
  bool jsonOutput{false};

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("input-file,i", po::value<std::string>(&inputFile)->required(), "input file name")
      ("nrdhs,n", po::value<unsigned int>(&nrdhs), "number of RDHs to go through")
      ("showRDHs,s",po::bool_switch(&showRDHs),"show RDHs")
      ("chargeSum,c",po::bool_switch(&chargeSum),"charge sum format")
      ("json,j",po::bool_switch(&jsonOutput),"output means and rms in json format")
      ("cru",po::value<uint16_t>(),"force cruId")
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  DumpOptions opt(nrdhs, showRDHs, jsonOutput);

  if (vm.count("cru")) {
    opt.cruId(vm["cru"].as<uint16_t>());
  }

  auto statChannel = rawdump(inputFile, opt);

  if (jsonOutput) {
    output(statChannel);
  }
  return 0;
}
