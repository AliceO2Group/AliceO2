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

/// \file digit2raw.cxx
/// \author ruben.shahoyan@cern.ch afurs@cern.ch

#include <boost/program_options.hpp>
#include <string>
#include "FT0Raw/RawWriterFT0.h"

/// MC->raw conversion for FT0

namespace bpo = boost::program_options;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n  " + std::string(argv[0]) +
                                       "Convert FT0 digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    opt_general.add_options()("help,h", "Print this help message");
    // config with FT0 defaults
    o2::fit::DigitToRawConfig::configureExecOptions(opt_general, "ft0digits.root", "alio2-cr1-flp200");
    // common part
    opt_all.add(opt_general).add(opt_hidden);
    bpo::store(bpo::command_line_parser(argc, argv).options(opt_all).positional(opt_pos).run(), vm);

    if (vm.count("help")) {
      std::cout << opt_general << std::endl;
      exit(0);
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
  const o2::fit::DigitToRawConfig cfg(vm);
  if (!cfg.mEnablePadding) {
    o2::fit::DigitToRawDevice<o2::ft0::RawWriterFT0>::digit2raw(cfg);
  } else {
    o2::fit::DigitToRawDevice<o2::ft0::RawWriterFT0_padded>::digit2raw(cfg);
  }
  o2::raw::HBFUtils::Instance().print();
  return 0;
}
