// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/QC/exe/raw-ul-checker.cxx
/// \brief  Compares the user logic output with the raw input
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 December 2019

#include <cstdint>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <vector>
#include <fmt/format.h>
#include "boost/program_options.hpp"
#include "MIDQC/UserLogicChecker.h"
#include "MIDRaw/ElectronicsDelay.h"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/RawFileReader.h"

namespace po = boost::program_options;

bool readFile(std::string filename, o2::mid::Decoder& decoder, std::vector<o2::mid::ROBoard>& data, std::vector<o2::mid::ROFRecord>& rofRecords, unsigned long int nHBFs)
{
  std::cout << "Reading " << filename << std::endl;
  o2::mid::RawFileReader rawFileReader;
  if (!rawFileReader.init(filename.c_str())) {
    std::cout << "Cannot initialize file reader with " << filename << std::endl;
    return false;
  }
  long int hbfsRead = 0;
  while (rawFileReader.readHB(true)) {
    decoder.process(rawFileReader.getData());
    rawFileReader.clear();
    size_t offset = data.size();
    data.insert(data.end(), decoder.getData().begin(), decoder.getData().end());
    for (auto& rof : decoder.getROFRecords()) {
      rofRecords.emplace_back(rof.interactionRecord, rof.eventType, rof.firstEntry + offset, rof.nEntries);
    }
    ++hbfsRead;
    if (hbfsRead == nHBFs) {
      break;
    }
  }
  if (data.empty()) {
    std::cout << "No data found in " << filename << std::endl;
    return false;
  }
  return true;
}

std::vector<std::string> split(const std::string& str, char delimiter = ',')
{
  std::vector<std::string> tokens;
  std::string token;
  std::istringstream tokenStream(str);
  while (std::getline(tokenStream, token, delimiter)) {
    tokens.push_back(token);
  }
  return tokens;
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");
  std::string ulFilenames, bareFilenames, feeIdConfigFilename, crateMasksFilename;
  std::string outFilename = "check_ul.txt";
  unsigned long int nHBFs = 0;

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("ul-filenames", po::value<std::string>(&ulFilenames),"Comma-separated input raw filenames with CRU User Logic")
          ("bare-filenames", po::value<std::string>(&bareFilenames),"Comma-separated input raw filenames with bare CRU")
          ("feeId-config-file", po::value<std::string>(&feeIdConfigFilename),"Filename with crate FEE ID correspondence")
          ("crate-masks-file", po::value<std::string>(&crateMasksFilename),"Filename with crate masks")
          ("electronics-delay-file", po::value<std::string>(),"Filename with electronics delay")
          ("outFilename", po::value<std::string>(&outFilename),"Output filename")
          ("nHBFs", po::value<unsigned long int>(&nHBFs),"Number of HBFs to test")
          ("full", po::value<bool>()->implicit_value(true),"Full check");


  po::options_description hidden("hidden options");
  hidden.add_options()
          ("input", po::value<std::vector<std::string>>(),"Input filename");
  // clang-format on

  po::options_description cmdline;
  cmdline.add(generic).add(hidden);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << "Usage: " << argv[0] << "\n";
    std::cout << generic << std::endl;
    return 2;
  }
  if (ulFilenames.empty() || bareFilenames.empty()) {
    std::cout << "Please specify ul-filenames and bare-filenames" << std::endl;
    return 1;
  }

  o2::mid::FEEIdConfig feeIdConfig;
  if (!feeIdConfigFilename.empty()) {
    feeIdConfig = o2::mid::FEEIdConfig(feeIdConfigFilename.c_str());
  }

  o2::mid::CrateMasks crateMasks;
  if (!crateMasksFilename.empty()) {
    crateMasks = o2::mid::CrateMasks(crateMasksFilename.c_str());
  }

  o2::mid::ElectronicsDelay electronicsDelay;
  if (vm.count("electronics-delay-file")) {
    o2::mid::ElectronicsDelay electronicsDelay = o2::mid::readElectronicsDelay(vm["electronics-delay-file"].as<std::string>().c_str());
  }

  auto bareDecoder = o2::mid::Decoder(true, true, electronicsDelay, crateMasks, feeIdConfig);
  auto ulDecoder = o2::mid::Decoder(true, false, electronicsDelay, crateMasks, feeIdConfig);

  std::vector<o2::mid::ROBoard> bareData, ulData;
  std::vector<o2::mid::ROFRecord> bareRofs, ulRofs;

  auto bareFnames = split(bareFilenames);
  auto ulFnames = split(ulFilenames);

  o2::mid::UserLogicChecker checker;

  for (auto& fname : bareFnames) {
    if (!readFile(fname, bareDecoder, bareData, bareRofs, nHBFs)) {
      return 3;
    }
  }

  for (auto& fname : ulFnames) {
    if (!readFile(fname, ulDecoder, ulData, ulRofs, bareFnames.size() * nHBFs)) {
      return 3;
    }
  }

  if (false) {
    // The orbit information in the UL is not correctly treated
    // This means that its orbit will always be 0, unlike the bare data orbit
    // Let us set the orbit of the raw data to 0 so that the results can be synchronized
    for (auto& rof : bareRofs) {
      rof.interactionRecord.orbit = 0;
    }

    for (auto& rof : ulRofs) {
      rof.interactionRecord.orbit = 0;
    }
  }

  std::ofstream outFile(outFilename.c_str());
  if (!outFile.is_open()) {
    std::cout << "Cannot open output file " << outFilename << std::endl;
    return 3;
  }

  if (checker.process(bareData, bareRofs, ulData, ulRofs, vm.count("full"))) {
    std::cout << "Everything ok!" << std::endl;
  } else {
    std::cout << "Problems found. See " << outFilename << " for details" << std::endl;
    outFile << checker.getDebugMessage() << std::endl;
  }
  outFile << checker.getSummary() << std::endl;
  outFile.close();

  return 0;
}
