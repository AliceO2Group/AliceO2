// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   MID/QC/exe/raw-checker.cxx
/// \brief  Raw data checker for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 December 2019

#include <iostream>
#include <fstream>
#include "boost/program_options.hpp"
#include "MIDRaw/FEEIdConfig.h"
#include "MIDRaw/CrateMasks.h"
#include "MIDRaw/Decoder.h"
#include "MIDRaw/RawFileReader.h"
#include "MIDQC/RawDataChecker.h"

namespace po = boost::program_options;

std::string getOutFilename(const char* inFilename, const char* outDir)
{
  std::string basename(inFilename);
  std::string fdir = "./";
  auto pos = basename.find_last_of("/");
  if (pos != std::string::npos) {
    basename.erase(0, pos + 1);
    fdir = inFilename;
    fdir.erase(pos);
  }
  basename.insert(0, "check_");
  basename += ".txt";
  std::string outputDir(outDir);
  if (outputDir.empty()) {
    outputDir = fdir;
  }
  if (outputDir.back() != '/') {
    outputDir += "/";
  }
  std::string outFilename = outputDir + basename;
  return outFilename;
}

template <typename GBTDECODER>
int process(po::variables_map& vm)
{
  std::vector<std::string> inputfiles{vm["input"].as<std::vector<std::string>>()};

  o2::mid::Decoder<GBTDECODER> decoder;
  o2::mid::RawDataChecker checker;
  if (vm.count("feeId-config-file")) {
    o2::mid::FEEIdConfig feeIdConfig(vm["feeId-config-file"].as<std::string>().c_str());
    decoder.setFeeIdConfig(feeIdConfig);
  }
  if (vm.count("crate-masks-file")) {
    o2::mid::CrateMasks crateMasks(vm["crate-masks-file"].as<std::string>().c_str());
    decoder.setCrateMasks(crateMasks);
    checker.init(crateMasks);
  } else {
    checker.init(o2::mid::CrateMasks());
  }
  decoder.init(true);

  auto nHBs = vm["nHBs"].as<unsigned long int>();
  auto nMaxErrors = vm["max-errors"].as<unsigned long int>();

  for (auto& filename : inputfiles) {
    o2::mid::RawFileReader rawFileReader;
    if (!rawFileReader.init(filename.c_str())) {
      return 2;
    }
    if (vm.count("custom-memory-size")) {
      rawFileReader.setCustomPayloadSize(vm["custom-memory-size"].as<uint16_t>());
    }
    std::string outFilename = getOutFilename(filename.c_str(), vm["output-dir"].as<std::string>().c_str());
    std::ofstream outFile(outFilename.c_str());
    if (!outFile.is_open()) {
      std::cout << "Error: cannot create " << outFilename << std::endl;
      return 2;
    }
    std::cout << "Writing output to: " << outFilename << " ..." << std::endl;

    std::vector<o2::mid::LocalBoardRO> data;
    std::vector<o2::mid::ROFRecord> rofRecords;
    std::vector<o2::mid::ROFRecord> hbRecords;

    checker.clear();
    unsigned long int iHB = 0;
    std::stringstream summary;
    while (rawFileReader.readHB(vm.count("only-closed-HBs") > 0)) {
      decoder.process(rawFileReader.getData());
      rawFileReader.clear();
      size_t offset = data.size();
      data.insert(data.end(), decoder.getData().begin(), decoder.getData().end());
      for (auto& rof : decoder.getROFRecords()) {
        rofRecords.emplace_back(rof.interactionRecord, rof.eventType, rof.firstEntry + offset, rof.nEntries);
      }
      o2::InteractionRecord hb(0, iHB);
      hbRecords.emplace_back(hb, o2::mid::EventType::Noise, offset, decoder.getData().size());
      ++iHB;
      if ((nHBs > 0 && iHB >= nHBs)) {
        break;
      }
      bool isComplete = true; //(vm.count("bare") && decoder.isComplete());
      // Partial check
      if (isComplete) {
        // The check assumes that we have all data corresponding to one event.
        // However this might not be true since we read one HB at  the time.
        // So we must test that the event was fully read before running the check.
        if (!checker.process(data, rofRecords, hbRecords)) {
          outFile << checker.getDebugMessage() << "\n";
        }
        data.clear();
        rofRecords.clear();
        hbRecords.clear();

        if (checker.getNEventsFaulty() >= nMaxErrors) {
          summary << "Too many errors found: abort check!\n";
          break;
        }
      }
    }
    // Check the remaining data
    if (data.size() > 0 && !checker.process(data, rofRecords, hbRecords)) {
      outFile << checker.getDebugMessage() << "\n";
    }
    summary << "Number of busy raised: " << checker.getNBusyRaised() << "\n";
    summary << "Fraction of faulty events: " << checker.getNEventsFaulty() << " / " << checker.getNEventsProcessed() << " = " << static_cast<double>(checker.getNEventsFaulty()) / ((checker.getNEventsProcessed() > 0) ? static_cast<double>(checker.getNEventsProcessed()) : 1.) << "\n";
    outFile << summary.str();
    std::cout << summary.str();

    outFile.close();
  }

  return 0;
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("nHBs", po::value<unsigned long int>()->default_value(0),"Number of HBs read")
          ("only-closed-HBs", po::value<bool>()->implicit_value(true),"Return only closed HBs")
          ("custom-memory-size", po::value<uint16_t>()->implicit_value(0x2000 - 0x100),"Ignore read RDH. Use custom memory size")
          ("max-errors", po::value<unsigned long int>()->default_value(10000),"Maximum number of errors before aborting")
          ("feeId-config-file", po::value<std::string>(),"Filename with crate FEE ID correspondence")
          ("crate-masks-file", po::value<std::string>(),"Filename with crate masks")
          ("output-dir", po::value<std::string>()->default_value(""),"Output directory")
          ("bare", po::value<bool>()->implicit_value(true),"Use bare decoder");


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

  if (vm.count("bare")) {
    return process<o2::mid::GBTBareDecoder>(vm);
  }
  return process<o2::mid::GBTUserLogicDecoder>(vm);
}
