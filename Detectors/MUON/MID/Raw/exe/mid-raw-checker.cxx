// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file   mid-raw-checker.cxx
/// \brief  CRU bare data checker for MID
/// \author Diego Stocco <Diego.Stocco at cern.ch>
/// \date   09 December 2019

#include <iostream>
#include <fstream>
#include "boost/program_options.hpp"
#include "MIDRaw/CRUBareDecoder.h"
#include "MIDRaw/RawFileReader.h"
#include "CRUBareDataChecker.h"

namespace po = boost::program_options;

o2::header::RAWDataHeader buildCustomRDH()
{
  o2::header::RAWDataHeader rdh;
  rdh.word1 |= 0x2000;
  rdh.word1 |= ((0x2000 - 0x100)) << 16;
  return rdh;
}

int main(int argc, char* argv[])
{
  po::variables_map vm;
  po::options_description generic("Generic options");
  unsigned long int nHBs = 0;
  bool onlyClosedHBs = false;
  bool ignoreRDH = true;

  // clang-format off
  generic.add_options()
          ("help", "produce help message")
          ("nHBs", po::value<unsigned long int>(&nHBs),"Number of HBs read")
          ("only-closed-HBs", po::value<bool>(&onlyClosedHBs)->implicit_value(true),"Return only closed HBs")
          ("ignore-RDH", po::value<bool>(&ignoreRDH)->implicit_value(true),"Ignore read RDH. Use custom one instead");


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

  o2::mid::CRUBareDecoder decoder;
  decoder.init(true);

  o2::mid::CRUBareDataChecker checker;

  unsigned long int iHB = 0;

  for (auto& filename : inputfiles) {
    o2::mid::RawFileReader<uint8_t> rawFileReader;
    std::string outFilename = filename;
    auto pos = outFilename.find_last_of("/");
    if (pos == std::string::npos) {
      pos = 0;
    } else {
      ++pos;
    }
    outFilename.insert(pos, "check_");
    outFilename += ".txt";
    std::ofstream outFile(outFilename.c_str());
    if (!rawFileReader.init(filename.c_str())) {
      return 2;
    }
    if (ignoreRDH) {
      rawFileReader.setCustomRDH(buildCustomRDH());
    }
    std::vector<o2::mid::LocalBoardRO> data;
    std::vector<o2::mid::ROFRecord> rofRecords;
    std::stringstream ss;
    bool isFirst = true;
    while (rawFileReader.getState() == 0) {
      rawFileReader.readHB(onlyClosedHBs);
      decoder.process(rawFileReader.getData());
      rawFileReader.clear();
      size_t offset = data.size();
      std::copy(decoder.getData().begin(), decoder.getData().end(), std::back_inserter(data));
      for (auto& rof : decoder.getROFRecords()) {
        rofRecords.emplace_back(rof.interactionRecord, rof.eventType, rof.firstEntry + offset, rof.nEntries);
      }
      ss << "  HB: " << iHB << "  (line: " << 512 * iHB + 1 << ")";
      if (decoder.isComplete()) {
        // The check assumes that we have all data corresponding to one event.
        // However this might not be true since we read one HB at the time.
        // So we must test that the event was fully read before running the check.
        if (!checker.process(data, rofRecords, isFirst)) {
          outFile << ss.str() << "\n";
          outFile << checker.getDebugMessage() << "\n";
        }
        isFirst = false;
        std::stringstream clean;
        ss.swap(clean);
        data.clear();
        rofRecords.clear();
      }
      ++iHB;
      if (nHBs > 0 && iHB >= nHBs) {
        break;
      }
    }
    outFile << "Fraction of faulty events: " << checker.getNBCsFaulty() << " / " << checker.getNBCsProcessed() << " = " << static_cast<double>(checker.getNBCsFaulty()) / static_cast<double>(checker.getNBCsProcessed());

    outFile.close();
  }

  return 0;
}
