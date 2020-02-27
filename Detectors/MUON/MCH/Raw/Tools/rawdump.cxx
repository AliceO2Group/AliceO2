// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Aphecetche

#include "DumpBuffer.h"
#include "Headers/RAWDataHeader.h"
#include "MCHRawCommon/DataFormats.h"
#include "MCHRawDecoder/Decoder.h"
#include "MCHRawElecMap/Mapper.h"
#include "boost/program_options.hpp"
#include <chrono>
#include <fmt/format.h>
#include <fstream>
#include <gsl/span>
#include <iostream>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <optional>
#include <cstdint>

namespace po = boost::program_options;

extern std::ostream& operator<<(std::ostream&, const o2::header::RAWDataHeaderV4&);

using namespace o2::mch::raw;
using RDHv4 = o2::header::RAWDataHeaderV4;

class DumpOptions
{
 public:
  DumpOptions(unsigned int deId, unsigned int maxNofRDHs, bool showRDHs, bool jsonOutput)
    : mDeId{deId}, mMaxNofRDHs{maxNofRDHs == 0 ? std::numeric_limits<unsigned int>::max() : maxNofRDHs}, mShowRDHs{showRDHs}, mJSON{jsonOutput} {}

  unsigned int deId() const
  {
    return mDeId;
  }
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
  unsigned int mDeId;
  unsigned int mMaxNofRDHs;
  bool mShowRDHs;
  bool mJSON;
  std::optional<uint16_t> mCruId{std::nullopt};
};

struct Stat {
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

std::ostream& operator<<(std::ostream& os, const Stat& s)
{
  os << fmt::format("MEAN {:7.3f} RMS {:7.3f} NSAMPLES {:5d} ", s.mean, s.rms, s.n);
  return os;
}
template <typename FORMAT, typename CHARGESUM, typename RDH>
std::map<std::string, Stat> rawdump(std::string input, DumpOptions opt)
{
  std::ifstream in(input.c_str(), std::ios::binary);
  if (!in.good()) {
    std::cout << "could not open file " << input << "\n";
    return {};
  }
  constexpr size_t pageSize = 8192;

  std::array<uint8_t, pageSize> buffer;
  gsl::span<uint8_t> sbuffer(buffer);

  size_t ndigits{0};

  std::map<std::string, int> uniqueDS;
  std::map<std::string, int> uniqueChannel;
  std::map<std::string, Stat> statChannel;

  memset(&buffer[0], 0, buffer.size());
  auto channelHandler = [&ndigits, &uniqueDS, &uniqueChannel, &statChannel](DsElecId dsId,
                                                                            uint8_t channel, o2::mch::raw::SampaCluster sc) {
    auto s = asString(dsId);
    uniqueDS[s]++;
    auto ch = fmt::format("{}-CH{}", s, channel);
    uniqueChannel[ch]++;
    auto& stat = statChannel[ch];
    for (auto d = 0; d < sc.nofSamples(); d++) {
      stat.incr(sc.samples[d]);
    }
    ++ndigits;
  };

  auto cruLink2solar = o2::mch::raw::createCruLink2SolarMapper<ElectronicMapperGenerated>();

  size_t nrdhs{0};
  auto rdhHandler = [&](const RDH& rdh) -> std::optional<RDH> {
    nrdhs++;
    if (opt.showRDHs()) {
      std::cout << nrdhs << "--" << rdh << "\n";
    }
    auto r = rdh;
    auto cruId = r.cruID;
    if (opt.cruId().has_value()) {
      // force cruId to externally given value
      cruId = opt.cruId().value();
    }
    auto linkId = rdhLinkId(r);
    auto solar = cruLink2solar(o2::mch::raw::CruLinkId(cruId, linkId, opt.deId()));
    if (!solar.has_value()) {
      std::cout << fmt::format("ERROR - Could not get solarUID from CRU,LINK=({},{},{})\n",
                               cruId, linkId, opt.deId());
      return std::nullopt;
    }
    r.feeId = solar.value();
    return r;
  };

  o2::mch::raw::Decoder decode = o2::mch::raw::createDecoder<FORMAT, CHARGESUM, RDH>(rdhHandler, channelHandler);

  std::vector<std::chrono::microseconds> timers;

  size_t npages{0};
  DecoderStat decStat;

  while (npages < opt.maxNofRDHs() && in.read(reinterpret_cast<char*>(&buffer[0]), pageSize)) {
    npages++;
    decStat = decode(sbuffer);
  }

  if (!opt.json()) {
    std::cout << ndigits << " digits seen - " << nrdhs << " RDHs seen - " << npages << " npages read\n";
    std::cout << "#unique DS=" << uniqueDS.size() << " #unique Channel=" << uniqueChannel.size() << "\n";
    std::cout << decStat << "\n";
  }
  return statChannel;
}

void output(const std::map<std::string, Stat>& channels)
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
  unsigned int deId{0};
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
      ("userLogic,u",po::bool_switch(&userLogic),"user logic format")
      ("chargeSum,c",po::bool_switch(&chargeSum),"charge sum format")
      ("json,j",po::bool_switch(&jsonOutput),"output means and rms in json format")
      ("de,d",po::value<unsigned int>(&deId)->required(),"detection element id of the data to be decoded")
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

  DumpOptions opt(deId, nrdhs, showRDHs, jsonOutput);
  std::map<std::string, Stat> statChannel;

  if (vm.count("cru")) {
    opt.cruId(vm["cru"].as<uint16_t>());
  }
  if (userLogic) {
    if (chargeSum) {
      statChannel = rawdump<UserLogicFormat, ChargeSumMode, RDHv4>(inputFile, opt);
    } else {
      statChannel = rawdump<UserLogicFormat, SampleMode, RDHv4>(inputFile, opt);
    }
  } else {
    if (chargeSum) {
      statChannel = rawdump<BareFormat, ChargeSumMode, RDHv4>(inputFile, opt);
    } else {
      statChannel = rawdump<BareFormat, SampleMode, RDHv4>(inputFile, opt);
    }
  }

  if (jsonOutput) {
    output(statChannel);
  }
  return 0;
}
