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

#include <catch_amalgamated.hpp>
#include <boost/program_options.hpp>
#include "../src/OptionsHelpers.h"
namespace bpo = boost::program_options;

TEST_CASE("Merging")
{
  boost::program_options::options_description desc1;
  desc1.add_options()("help,h", bpo::value<std::string>()->default_value("foo"), "Print help message")("help,h", bpo::value<std::string>()->default_value("foo"), "Print help message");
  auto res = o2::framework::OptionsHelpers::makeUniqueOptions(desc1);
  REQUIRE(res.options().size() == 1);
}
