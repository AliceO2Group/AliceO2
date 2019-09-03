// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file testInfrastructureBuilder.cxx
/// \brief A unit test of mergers
///
/// \author Piotr Konopka, piotr.jan.konopka@cern.ch

#define BOOST_TEST_MODULE Test Utilities Mergers
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Mergers/MergerInfrastructureBuilder.h"

#include "Framework/DataSpecUtils.h"
#include <iostream>

using namespace o2::framework;
using namespace o2::experimental::mergers;

BOOST_AUTO_TEST_CASE(InfrastructureBuilderUnconfigured)
{
  MergerInfrastructureBuilder builder;

  // unconfigured builder
  BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);

  // partially configured builder
  builder.setInfrastructureName("name");
  builder.setInputSpecs({{"one", "TST", "test", 1}});
  BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);

  builder.setOutputSpec({{"main"}, "TST", "test", 0});
  builder.setInfrastructureName("");
  BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);

  builder.setInfrastructureName("name");
  builder.setInputSpecs({});
  BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);

  // configured builder (no exception)
  builder.setInputSpecs({{"one", "TST", "test", 1}});
  BOOST_CHECK_NO_THROW(builder.generateInfrastructure());
}

BOOST_AUTO_TEST_CASE(InfrastructureBuilderLayers)
{
  MergerInfrastructureBuilder builder;
  builder.setInfrastructureName("name");
  builder.setInputSpecs({{"one", "TST", "test", 1},
                         {"two", "TST", "test", 2},
                         {"thr", "TST", "test", 3},
                         {"fou", "TST", "test", 4},
                         {"fiv", "TST", "test", 5},
                         {"six", "TST", "test", 6},
                         {"sev", "TST", "test", 7}});
  builder.setOutputSpec({{"main"}, "TST", "test", 0});
  MergerConfig config;

  {
    config.topologySize = {TopologySize::NumberOfLayers, 0};
    builder.setConfig(config);
    BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);
  }

  {
    config.topologySize = {TopologySize::NumberOfLayers, 1};
    builder.setConfig(config);
    auto mergersTopology = builder.generateInfrastructure();

    BOOST_REQUIRE_EQUAL(mergersTopology.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[0].inputs.size(), 7);
    BOOST_REQUIRE_EQUAL(mergersTopology[0].outputs.size(), 1);

    auto concrete = DataSpecUtils::asConcreteDataMatcher(mergersTopology[0].outputs[0]);
    BOOST_CHECK_EQUAL(concrete.origin.str, "TST");
    BOOST_CHECK_EQUAL(concrete.description.str, "test");
    BOOST_CHECK_EQUAL(concrete.subSpec, 0);
  }

  {
    config.topologySize = {TopologySize::NumberOfLayers, 2};
    builder.setConfig(config);
    auto mergersTopology = builder.generateInfrastructure();

    BOOST_REQUIRE_EQUAL(mergersTopology.size(), 4);

    // the first layer
    BOOST_CHECK_EQUAL(mergersTopology[0].inputs.size(), 3);
    BOOST_CHECK_EQUAL(mergersTopology[0].outputs.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[1].inputs.size(), 2);
    BOOST_CHECK_EQUAL(mergersTopology[1].outputs.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[2].inputs.size(), 2);
    BOOST_CHECK_EQUAL(mergersTopology[2].outputs.size(), 1);

    // the second layer
    BOOST_CHECK_EQUAL(mergersTopology[3].inputs.size(), 3);
    BOOST_CHECK_EQUAL(mergersTopology[3].outputs.size(), 1);
    auto concrete = DataSpecUtils::asConcreteDataMatcher(mergersTopology[3].outputs[0]);
    BOOST_REQUIRE_EQUAL(concrete.origin.str, "TST");
    BOOST_CHECK_EQUAL(concrete.description.str, "test");
    BOOST_CHECK_EQUAL(concrete.subSpec, 0);
  }
}

BOOST_AUTO_TEST_CASE(InfrastructureBuilderReductionFactor)
{
  MergerInfrastructureBuilder builder;
  builder.setInfrastructureName("name");
  builder.setInputSpecs({{"one", "TST", "test", 1},
                         {"two", "TST", "test", 2},
                         {"thr", "TST", "test", 3},
                         {"fou", "TST", "test", 4},
                         {"fiv", "TST", "test", 5},
                         {"six", "TST", "test", 6},
                         {"sev", "TST", "test", 7}});
  builder.setOutputSpec({{"main"}, "TST", "test", 0});
  MergerConfig config;

  {
    config.topologySize = {TopologySize::ReductionFactor, 1};
    builder.setConfig(config);
    BOOST_CHECK_THROW(builder.generateInfrastructure(), std::runtime_error);
  }

  {
    config.topologySize = {TopologySize::ReductionFactor, 7};
    builder.setConfig(config);
    auto mergersTopology = builder.generateInfrastructure();

    BOOST_REQUIRE_EQUAL(mergersTopology.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[0].inputs.size(), 7);
    BOOST_REQUIRE_EQUAL(mergersTopology[0].outputs.size(), 1);

    auto concrete = DataSpecUtils::asConcreteDataMatcher(mergersTopology[0].outputs[0]);
    BOOST_CHECK_EQUAL(concrete.origin.str, "TST");
    BOOST_CHECK_EQUAL(concrete.description.str, "test");
    BOOST_CHECK_EQUAL(concrete.subSpec, 0);
  }

  {
    config.topologySize = {TopologySize::ReductionFactor, 3};
    builder.setConfig(config);
    auto mergersTopology = builder.generateInfrastructure();

    BOOST_REQUIRE_EQUAL(mergersTopology.size(), 4);

    // the first layer
    BOOST_CHECK_EQUAL(mergersTopology[0].inputs.size(), 3);
    BOOST_CHECK_EQUAL(mergersTopology[0].outputs.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[1].inputs.size(), 2);
    BOOST_CHECK_EQUAL(mergersTopology[1].outputs.size(), 1);
    BOOST_CHECK_EQUAL(mergersTopology[2].inputs.size(), 2);
    BOOST_CHECK_EQUAL(mergersTopology[2].outputs.size(), 1);

    // the second layer
    BOOST_CHECK_EQUAL(mergersTopology[3].inputs.size(), 3);
    BOOST_CHECK_EQUAL(mergersTopology[3].outputs.size(), 1);

    auto concrete = DataSpecUtils::asConcreteDataMatcher(mergersTopology[3].outputs[0]);
    BOOST_REQUIRE_EQUAL(concrete.origin.str, "TST");
    BOOST_CHECK_EQUAL(concrete.description.str, "test");
    BOOST_CHECK_EQUAL(concrete.subSpec, 0);
  }
}
