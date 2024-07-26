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

/// \file ITSDataSimulator.cxx
/// \author knaumov@cern.ch

#include "DataFormatsITSMFT/Digit.h"
#include "ITSSimulation/ITSDataSimulator.h"

#include <boost/program_options.hpp>
#include <iostream>
#include <string>

namespace bpo = boost::program_options;
using namespace o2::itsmft;

std::vector<PixelData> ITSDataSimulator::generateChipData()
{
  std::vector<PixelData> vec;
  uint32_t numOfPixels = rand() % mMaxPixelsPerChip;
  while (vec.size() < numOfPixels) {
    int row = rand() % SegmentationAlpide::NRows;
    int col = rand() % SegmentationAlpide::NCols;
    PixelData pixel(row, col);
    vec.push_back(pixel);
  }
  std::sort(vec.begin(), vec.end());
  if (!mDoErrors) {
    // If errors are disabled, chips should not contain
    // pixels fired multiple times
    auto iter = std::unique(vec.begin(), vec.end());
    vec.erase(iter, vec.end());
  }
  return vec;
}

void ITSDataSimulator::simulate()
{
  // Generate the chip data
  std::map<uint32_t, std::vector<PixelData>> chipData;
  while (chipData.size() < mNumberOfChips) {
    uint32_t chipID = rand() % MaxChipID;
    if (!chipData.contains(chipID)) {
      chipData.emplace(chipID, generateChipData());
    }
  }

  if (mDoDigits) {
    std::vector<Digit> digVec;
    for (auto const& chip : chipData) {
      uint32_t chipID = chip.first;
      const std::vector<PixelData>& pixels = chip.second;
      for (auto pixel : pixels) {
        Digit dig(chipID, pixel.getRow(), pixel.getCol());
        digVec.push_back(dig);
      }
    }
    // TODO: Save the digits to a file
  }
}

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Simulates ALPIDE data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("verbosity,v", bpo::value<uint32_t>()->default_value(0),
               "verbosity level [0 = no output]");
    add_option("digits",
               bpo::value<bool>()->default_value(false)->implicit_value(true),
               "generate the data in the digits format");
    add_option("enable-errors",
               bpo::value<bool>()->default_value(false)->implicit_value(true),
               "enable additon of errors to the raw data");
    add_option("seed", bpo::value<int32_t>()->default_value(0),
               "random seed for data generation");
    add_option(
      "max-pixels-per-chip", bpo::value<uint32_t>()->default_value(100),
      ("maximum number of fired pixels per chip (0 - " +
       std::to_string(ITSDataSimulator::MaxPixelsPerChip) +
       ")")
        .c_str());
    add_option("number-of-chip", bpo::value<uint32_t>()->default_value(10),
               ("number of chips to be present in the data (0 - " +
                std::to_string(ITSDataSimulator::MaxChipID) + ")")
                 .c_str());
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""),
               "comma-separated configKeyValues");

    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv)
                 .options(opt_all)
                 .positional(opt_pos)
                 .run(),
               vm);

    if (vm.count("help")) {
      std::cout << opt_general << std::endl;
      exit(0);
    }

    if (vm["max-pixels-per-chip"].as<uint32_t>() >
        ITSDataSimulator::MaxPixelsPerChip) {
      std::cerr << "Invalid max pixels per chip, valid range (0, "
                << ITSDataSimulator::MaxPixelsPerChip << ")" << std::endl;
      exit(1);
    }

    if (vm["number-of-chip"].as<uint32_t>() > ITSDataSimulator::MaxChipID) {
      std::cerr << "Invalid number of chips, valid range (0, "
                << ITSDataSimulator::MaxChipID << ")" << std::endl;
      exit(1);
    }

    bpo::notify(vm);
  } catch (bpo::error& e) {
    std::cerr << "ERROR: " << e.what() << std::endl
              << std::endl;
    std::cerr << opt_general << std::endl;
    exit(1);
  } catch (std::exception& e) {
    std::cerr << e.what() << ", application will now exit" << std::endl;
    exit(2);
  }

  ITSDataSimulator simulator(
    vm["seed"].as<int32_t>(), vm["number-of-chip"].as<uint32_t>(),
    vm["max-pixels-per-chip"].as<uint32_t>(), vm["digits"].as<bool>(),
    vm["enable-errors"].as<bool>());

  simulator.simulate();
}
