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

/// @author  Laurent Aphecetche

#include "DataFormatsMCH/DsChannelId.h"
#include "MCHConditions/DCSAliases.h"
#include "MCHGlobalMapping/Mapper.h"
#include "MCHRawElecMap/DsDetId.h"
#include "MCHRawElecMap/DsElecId.h"
#include "MCHRawElecMap/Mapper.h"
#include "boost/program_options.hpp"
#include <cstdint>
#include <fmt/format.h>
#include <fstream>
#include <gsl/span>
#include <iostream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <rapidjson/document.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <stdexcept>
#include <string>

namespace po = boost::program_options;
using namespace o2::mch::raw;

uint16_t computeDsBinX(int feeId, int linkId, int elinkId)
{
  constexpr uint64_t maxLinkId = 12;
  constexpr uint64_t maxElinkId = 40;

  int v = feeId * maxLinkId * maxElinkId +
          (linkId % maxLinkId) * maxElinkId + elinkId + 1;
  return static_cast<uint16_t>(v & 0xFFFF);
}

struct DualSampaInfo {
  uint16_t dsBin;
  uint16_t dsBinX;
  int deId;
  int dsId;
  uint16_t feeId;
  uint8_t linkId;
  uint8_t eLinkId;
  uint16_t solarId;
  uint8_t nch;
  uint32_t firstDsChannelId;
};

std::vector<DualSampaInfo> dualSampaInfos;
std::vector<DualSampaInfo> computeDualSampaInfos();

using namespace std::literals;

std::string stripAlias(std::string alias)
{
  std::vector<std::string> remove = {".vMon", ".iMon", "di.SenseVoltage", "an.SenseVoltage"};

  for (auto r : remove) {
    if (alias.find(r) != std::string::npos) {
      alias.replace(alias.find(r), r.size(), "");
    }
  }
  return alias;
}

void dcs2json()
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  auto dualSampaInfos = computeDualSampaInfos();

  auto aliases = o2::mch::dcs::aliases({o2::mch::dcs::MeasurementType::HV_I,
                                        o2::mch::dcs::MeasurementType::LV_V_FEE_ANALOG,
                                        o2::mch::dcs::MeasurementType::LV_V_SOLAR});

  std::map<std::string, std::set<int>> dsIndicesPerAliasBase;

  for (auto alias : aliases) {
    auto s = stripAlias(alias);
    auto indices = o2::mch::dcs::aliasToDsIndices(alias);
    dsIndicesPerAliasBase.emplace(s, indices);
  }

  writer.StartArray();
  for (auto p : dsIndicesPerAliasBase) {
    writer.StartObject();
    writer.Key("alias");
    auto s = p.first;
    writer.String(s.c_str());
    writer.Key("ds");
    writer.StartArray();
    for (auto i : p.second) {
      for (const auto& dsi : dualSampaInfos) {
        if (dsi.dsBin == i) {
          writer.Int(dsi.dsBinX);
        }
      }
    }
    writer.EndArray();
    writer.EndObject();
  }
  writer.EndArray();

  std::cout << buffer.GetString() << "\n";
}

std::vector<DualSampaInfo> computeDualSampaInfos()
{
  uint16_t dsBin{0};

  auto elec2det = createElec2DetMapper<ElectronicMapperGenerated>();
  auto det2elec = createDet2ElecMapper<ElectronicMapperGenerated>();
  auto solar2FeeLink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();

  for (uint16_t dsIndex = 0; dsIndex < o2::mch::NumberOfDualSampas; ++dsIndex) {
    DsDetId det{o2::mch::getDsDetId(dsIndex)};
    auto elec = det2elec(det);
    if (!elec.has_value()) {
      throw std::runtime_error("mapping is wrong somewhere...");
    }
    auto eLinkId = elec->elinkId();
    auto solarId = elec->solarId();
    auto s2f = solar2FeeLink(solarId);
    if (!s2f.has_value()) {
      throw std::runtime_error("mapping is wrong somewhere...");
    }
    auto feeId = s2f->feeId();
    auto linkId = s2f->linkId();

    auto dsBinX = computeDsBinX(feeId, linkId, eLinkId);

    uint8_t nch = o2::mch::numberOfDualSampaChannels(dsIndex);

    o2::mch::DsChannelId firstDsChannelId(solarId, eLinkId, 0);

    auto dsId = det.dsId();
    auto deId = det.deId();

    dualSampaInfos.emplace_back(DualSampaInfo{dsIndex, dsBinX,
                                              deId, dsId,
                                              feeId, linkId,
                                              eLinkId, solarId, nch,
                                              firstDsChannelId.value()});
  }
  return dualSampaInfos;
}

void solar2json(bool mchview)
{
  rapidjson::StringBuffer buffer;
  rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);

  auto solar2FeeLink = createSolar2FeeLinkMapper<ElectronicMapperGenerated>();
  auto dualSampaInfos = computeDualSampaInfos();

  auto outputSolars = [&](uint16_t solarId) {
    auto dualSampas = getDualSampas<ElectronicMapperGenerated>(solarId);
    auto deId = dualSampas.begin()->deId();

    writer.StartObject();
    writer.Key("solarId");
    writer.Int(solarId);
    auto feelink = solar2FeeLink(solarId);
    uint16_t feeId{std::numeric_limits<uint16_t>::max()};
    uint8_t linkId{std::numeric_limits<uint8_t>::max()};
    if (feelink.has_value()) {
      feeId = feelink->feeId();
      linkId = feelink->linkId();
      writer.Key("FeeId");
      writer.Int(feeId);
      writer.Key("LinkId");
      writer.Int(linkId);
    }
    writer.Key("deId");
    writer.Int(deId);
    writer.Key("ds");
    writer.StartArray();
    for (const auto& dsi : dualSampaInfos) {
      if (dsi.feeId == feeId &&
          dsi.linkId == linkId) {
        writer.StartObject();
        writer.Key("dsId");
        writer.Int(dsi.dsId);
        if (mchview) {
          writer.Key("elinkId");
        } else {
          writer.Key("eLinkId");
        }
        writer.Int(dsi.eLinkId);
        if (dsi.nch != 64) {
          writer.Key("nch");
          writer.Int(dsi.nch);
        }
        if (mchview) {
          writer.Key("dsbin");
          writer.Int(dsi.dsBinX);
        } else {
          writer.Key("binX");
          writer.Int(dsi.dsBinX);
          writer.Key("bin");
          writer.Int(dsi.dsBin);
          writer.Key("fdci");
          writer.Int(dsi.firstDsChannelId);
        }
        writer.EndObject();
      }
    }
    writer.EndArray();
    writer.EndObject();
  };

  auto solarIds = getSolarUIDs<ElectronicMapperGenerated>();
  writer.StartArray();
  for (auto s : solarIds) {
    outputSolars(s);
  }
  writer.EndArray();

  std::cout << buffer.GetString() << "\n";
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
      ("help,h", "produce help message")
      ("solar,s","output solar based file")
      ("mchview,m","output format suitable for mchview")
      ("dcs,d","output dcs aliases -> dual sampa indices file")
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

  if (vm.count("solar")) {
    solar2json(vm.count("mchview"));
  }

  if (vm.count("dcs")) {
    dcs2json();
  }
  return 0;
}
