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

#include "boost/program_options.hpp"
#include <iostream>
#include <fstream>
#include "DigitSampler.h"
#include "DigitSink.h"
#include "DataFormatsMCH/Digit.h"
#include "DataFormatsMCH/ROFRecord.h"
#include <fmt/format.h>
#include "ProgOptions.h"

namespace po = boost::program_options;

/**
 * o2-mch-digits-file-dumper is a small helper program to inspect
 * MCH digit binary files (the ones that are not in CTF format)
 */

int main(int argc, char* argv[])
{
  std::string inputFile;
  po::variables_map vm;
  po::options_description options("options");
  bool count{false};
  bool describe{false};
  bool printDigits{false};
  bool printTFs{false};
  int maxNofTimeFrames{std::numeric_limits<int>::max()};
  int firstTimeFrame{0};

  // clang-format off
  // clang-format off
  options.add_options()
      ("help,h", "produce help message")
      ("infile,i", po::value<std::string>(&inputFile)->required(), "input file name")
      ("count,c",po::bool_switch(&count),"count items (rofs, tfs, etc...)")
      ("describe,d",po::bool_switch(&describe),"describe file format")
      (OPTNAME_PRINT_DIGITS,po::bool_switch(&printDigits),OPTHELP_PRINT_DIGITS)
      (OPTNAME_PRINT_TFS,po::bool_switch(&printTFs),OPTHELP_PRINT_TFS)
      (OPTNAME_MAX_NOF_TFS,po::value<int>(&maxNofTimeFrames),OPTHELP_MAX_NOF_TFS)
      (OPTNAME_FIRST_TF,po::value<int>(&firstTimeFrame),OPTHELP_FIRST_TF)
      ;
  // clang-format on

  po::options_description cmdline;
  cmdline.add(options);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);

  if (vm.count("help")) {
    std::cout << options << "\n";
    return 2;
  }

  try {
    po::notify(vm);
  } catch (boost::program_options::error& e) {
    std::cout << "Error: " << e.what() << "\n";
    exit(1);
  }

  std::ifstream in(inputFile.c_str());
  if (!in.is_open()) {
    std::cerr << "cannot open input file " << inputFile << "\n";
    return 3;
  }
  o2::mch::io::DigitSampler dr(in);

  if (describe) {
    std::cout << dr.fileFormat() << "\n";
  }
  if (count) {
    std::cout << fmt::format("nTFs {} nROFs {} nDigits {}\n",
                             dr.nofTimeFrames(), dr.nofROFs(), dr.nofDigits());
  }

  if (printTFs || printDigits) {
    int tfid{0};
    int tfcount{0};
    o2::mch::io::DigitSink dump(std::cout);
    std::vector<o2::mch::Digit> digits;
    std::vector<o2::mch::ROFRecord> rofs;
    while (dr.read(digits, rofs)) {
      if (tfid >= firstTimeFrame && tfcount < maxNofTimeFrames) {
        if (printTFs) {
          std::cout << fmt::format("TF {:5d} {:4d} rofs {:5d} digits\n",
                                   tfid, rofs.size(), digits.size());
        }
        if (printDigits) {
          dump.write(digits, rofs);
        }
        ++tfcount;
      }
      ++tfid;
    }
  }
  return 0;
}
