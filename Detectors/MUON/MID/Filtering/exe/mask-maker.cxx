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

/// \file   MID/Filtering/exe/mask-maker.cxx
/// \brief  Utility to build masks from data
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   12 March 2020

#include <iostream>
#include <vector>
#include "boost/program_options.hpp"
#include "CommonDataFormat/InteractionRecord.h"
#include "DataFormatsMID/ROBoard.h"
#include "DataFormatsMID/ROFRecord.h"
#include "MIDRaw/ColumnDataToLocalBoard.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/DecodedDataAggregator.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/RawFileReader.h"
#include "MIDRaw/ROBoardConfigHandler.h"
#include "MIDFiltering/ChannelMasksHandler.h"
#include "MIDFiltering/ChannelScalers.h"
#include "MIDFiltering/FetToDead.h"
#include "MIDFiltering/MaskMaker.h"

namespace po = boost::program_options;

bool processScalers(const o2::mid::ChannelScalers& scalers, unsigned long nEvents, double threshold, const std::vector<o2::mid::ColumnData>& refMasks, const char* outFilename)
{
  auto masks = o2::mid::makeMasks(scalers, nEvents, threshold, refMasks);
  if (masks.empty()) {
    std::cout << "No problematic digit found" << std::endl;
    return true;
  }

  std::cout << "Problematic digits found. Corresponding masks:" << std::endl;
  for (auto& mask : masks) {
    std::cout << mask << std::endl;
  }

  o2::mid::ColumnDataToLocalBoard colToBoard;
  o2::mid::ROBoardConfigHandler roBoardCfgHandler;

  std::cout << "\nCorresponding boards masks:" << std::endl;
  colToBoard.process(masks);
  auto roMasks = colToBoard.getData();
  for (auto& board : roMasks) {
    std::cout << board << std::endl;
  }
  std::cout << "\nMask file produced: " << outFilename << std::endl;
  o2::mid::ChannelMasksHandler masksHandler;
  masksHandler.setFromChannelMasks(masks);
  masksHandler.merge(o2::mid::makeDefaultMasks());
  colToBoard.process(masksHandler.getMasks());
  roBoardCfgHandler.updateMasks(colToBoard.getData());
  roBoardCfgHandler.write(outFilename);
  return false;
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("feeId-config-file", po::value<std::string>(),"Filename with crate FEE ID correspondence")
          ("crate-masks-file", po::value<std::string>(),"Filename with crate masks")
          ("electronics-delay-file", po::value<std::string>(),"Filename with electronics delay")
          ("occupancy-threshold", po::value<double>()->default_value(0.9),"Occupancy threshold");


  po::options_description hidden("hidden options");
  hidden.add_options()
          ("input", po::value<std::vector<std::string>>(),"Input filename");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic).add(hidden);

  po::positional_options_description pos;
  pos.add("input", -1);

  po::store(po::command_line_parser(argc, argv).options(cmdline).positional(pos).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage: " << argv[0] << " <input_raw_filename> [input_raw_filename_1 ...]\n";
    std::cout << generic << std::endl;
    return 2;
  }
  if (vm.count("input") == 0) {
    std::cout << "no input file specified" << std::endl;
    return 1;
  }

  std::vector<std::string> inputfiles{vm["input"].as<std::vector<std::string>>()};

  std::unique_ptr<o2::mid::Decoder> decoder{nullptr};

  o2::mid::FEEIdConfig feeIdConfig;
  if (vm.count("feeId-config-file")) {
    feeIdConfig = o2::mid::FEEIdConfig(vm["feeId-config-file"].as<std::string>().c_str());
  }

  o2::mid::ElectronicsDelay electronicsDelay;
  if (vm.count("electronics-delay-file")) {
    electronicsDelay = o2::mid::readElectronicsDelay(vm["electronics-delay-file"].as<std::string>().c_str());
  }

  o2::mid::CrateMasks crateMasks;
  if (vm.count("crate-masks-file")) {
    crateMasks = o2::mid::CrateMasks(vm["crate-masks-file"].as<std::string>().c_str());
  }

  auto threshold = vm["occupancy-threshold"].as<double>();

  std::vector<o2::mid::ROBoard> data;
  std::vector<o2::mid::ROFRecord> rofRecords;

  for (auto& filename : inputfiles) {
    o2::mid::RawFileReader rawFileReader;
    if (!rawFileReader.init(filename.c_str())) {
      return 2;
    }

    while (rawFileReader.readHB() > 0) {
      if (!decoder) {
        auto const* rdhPtr = reinterpret_cast<const o2::header::RDHAny*>(rawFileReader.getData().data());
        decoder = o2::mid::createDecoder(*rdhPtr, false, electronicsDelay, crateMasks, feeIdConfig);
      }
      decoder->process(rawFileReader.getData());
      rawFileReader.clear();
      size_t offset = data.size();
      data.insert(data.end(), decoder->getData().begin(), decoder->getData().end());
      for (auto& rof : decoder->getROFRecords()) {
        rofRecords.emplace_back(rof.interactionRecord, rof.eventType, rof.firstEntry + offset, rof.nEntries);
      }
    }
  }

  o2::mid::DecodedDataAggregator aggregator;
  aggregator.process(data, rofRecords);
  std::array<o2::mid::ChannelScalers, 2> scalers;
  for (auto& noisy : aggregator.getData(o2::mid::EventType::Calib)) {
    scalers[0].count(noisy);
  }

  unsigned long nEvents = aggregator.getROFRecords(o2::mid::EventType::Calib).size();
  auto refMasks = makeDefaultMasksFromCrateConfig(feeIdConfig, crateMasks);

  bool isOk = true;
  std::cout << "\nCHECKING NOISY CHANNELS:" << std::endl;
  isOk &= processScalers(scalers[0], nEvents, threshold, refMasks, "calib_mask.txt");

  o2::mid::FetToDead fetToDead;
  fetToDead.setMasks(refMasks);
  auto fetRofs = aggregator.getROFRecords(o2::mid::EventType::FET);
  auto fets = aggregator.getData(o2::mid::EventType::FET);

  for (auto& rof : fetRofs) {
    std::vector<o2::mid::ColumnData> eventFets(fets.begin() + rof.firstEntry, fets.begin() + rof.getEndIndex());
    auto deadChannels = fetToDead.process(eventFets);
    for (auto& dead : deadChannels) {
      scalers[1].count(dead);
    }
  }
  std::cout << "\nCHECKING DEAD CHANNELS:" << std::endl;
  isOk &= processScalers(scalers[1], nEvents, threshold, refMasks, "FET_mask.txt");

  return isOk ? 0 : 1;
}
