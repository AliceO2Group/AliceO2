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
#include <fairmq/ProgOptions.h>
#include "Framework/FairOptionsRetriever.h"
#include "Framework/RootConfigParamHelpers.h"
#include "Framework/ConfigParamRegistry.h"
#include <boost/program_options.hpp>
#include "TestClasses.h"

using namespace o2::framework;
namespace bpo = boost::program_options;

TEST_CASE("TestRootConfigParamRegistry")
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                             //
    ("foo.x", bpo::value<int>()->default_value(2))      //
    ("foo.y", bpo::value<float>()->default_value(3.f)); //

  fair::mq::ProgOptions* options = new fair::mq::ProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd",
                     "--foo.x", "1",
                     "--foo.y", "2"},
                    true);

  std::vector<ConfigParamSpec> specs = RootConfigParamHelpers::asConfigParamSpecs<o2::test::SimplePODClass>("foo");

  std::vector<std::unique_ptr<ParamRetriever>> retrievers;
  std::unique_ptr<ParamRetriever> fairmqRetriver{new FairOptionsRetriever(options)};
  retrievers.emplace_back(std::move(fairmqRetriver));

  auto store = std::make_unique<ConfigParamStore>(specs, std::move(retrievers));
  store->preload();
  store->activate();
  ConfigParamRegistry registry(std::move(store));

  REQUIRE(registry.get<int>("foo.x") == 1);
  REQUIRE(registry.get<float>("foo.y") == 2.f);

  // We can get nested objects also via their top-level ptree.
  auto pt = registry.get<boost::property_tree::ptree>("foo");
  REQUIRE(pt.get<int>("x") == 1);
  REQUIRE(pt.get<float>("y") == 2.f);

  // And we can get it as a generic object as well.
  auto obj = RootConfigParamHelpers::as<o2::test::SimplePODClass>(pt);
  REQUIRE(obj.x == 1);
  REQUIRE(obj.y == 2.f);
}
