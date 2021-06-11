// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file digi2raw.cxx
/// \breif an alias to o2-tof-reco-workflow -b --output-type raw --tof-raw-outdir <outdir>
/// \author ruben.shahoyan@cern.ch

#include <boost/program_options.hpp>
#include <iostream>
#include "CommonUtils/StringUtils.h"

namespace bpo = boost::program_options;

int main(int argc, char** argv)
{
  bpo::variables_map vm;
  bpo::options_description opt_general("Usage:\n" + std::string(argv[0]) +
                                       "Convert TOF digits to CRU raw data\n");
  bpo::options_description opt_hidden("");
  bpo::options_description opt_all;
  bpo::positional_options_description opt_pos;

  try {
    auto add_option = opt_general.add_options();
    add_option("help,h", "Print this help message");
    add_option("output-dir,o", bpo::value<std::string>()->default_value("./"), "output directory for raw data");
    add_option("file-for,f", bpo::value<std::string>()->default_value("cru"), "single file per: all,cru,link");
    add_option("configKeyValues", bpo::value<std::string>()->default_value(""), "comma-separated configKeyValues");
    //add_option("hbfutils-config,u", bpo::value<std::string>()->default_value(std::string(o2::base::NameConf::DIGITIZATIONCONFIGFILE)), "config file for HBFUtils (or none)");
    add_option("hbfutils-config,u", bpo::value<std::string>()->default_value("none"), "config file for HBFUtils (or none)");
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

  auto cmd = o2::utils::Str::concat_string("o2-tof-reco-workflow -b --output-type raw --tof-raw-outdir ", vm["output-dir"].as<std::string>(),
                                           " --tof-raw-file-for ", vm["file-for"].as<std::string>(),
                                           " --hbfutils-config ", vm["hbfutils-config"].as<std::string>(),
                                           R"( --configKeyValues ")", vm["configKeyValues"].as<std::string>(), '"');
  return system(cmd.c_str());
}
