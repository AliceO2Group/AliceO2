// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework ConfigParamRegistry
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <fairmq/options/FairMQProgOptions.h>
#include "Framework/FairOptionsRetriever.h"
#include "Framework/RootConfigParamHelpers.h"
#include "Framework/ConfigParamRegistry.h"
#include <boost/program_options.hpp>
#include "TestClasses.h"

using namespace o2::framework;
namespace bpo = boost::program_options;

BOOST_AUTO_TEST_CASE(TestConfigParamRegistry)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                             //
    ("foo.x", bpo::value<int>()->default_value(2))      //
    ("foo.y", bpo::value<float>()->default_value(3.f)); //

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({"cmd",
                     "--foo.x", "1",
                     "--foo.y", "2"},
                    false);

  std::vector<ConfigParamSpec> specs = RootConfigParamHelpers::asConfigParamSpecs<o2::test::SimplePODClass>("foo");

  auto retriever = std::make_unique<FairOptionsRetriever>(specs, options);
  ConfigParamRegistry registry(std::move(retriever));

  BOOST_CHECK_EQUAL(registry.get<int>("foo.x"), 1);
  BOOST_CHECK_EQUAL(registry.get<float>("foo.y"), 2.f);

  // We can get nested objects also via their top-level ptree.
  auto pt = registry.get<boost::property_tree::ptree>("foo");
  BOOST_CHECK_EQUAL(pt.get<int>("x"), 1);
  BOOST_CHECK_EQUAL(pt.get<float>("y"), 2.f);

  // And we can get it as a generic object as well.
  auto obj = RootConfigParamHelpers::as<o2::test::SimplePODClass>(pt);
  BOOST_CHECK_EQUAL(obj.x, 1);
  BOOST_CHECK_EQUAL(obj.y, 2.f);
}
