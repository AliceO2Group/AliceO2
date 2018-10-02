// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework OptionsRetriever
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>
#include <boost/program_options.hpp>
#include "Framework/FairOptionsRetriever.h"
#include <fairmq/options/FairMQProgOptions.h>

namespace bpo = boost::program_options;
using namespace o2::framework;

BOOST_AUTO_TEST_CASE(TestOptionsRetriever)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"));

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({ "cmd", "--aFloat", "1.0", "--aDouble", "2.0", "--anInt", "10", "--aBoolean", "--aString", "somethingelse" }, false);
  FairOptionsRetriever retriever(options);
  BOOST_CHECK_EQUAL(retriever.getFloat("aFloat"), 1.0);
  BOOST_CHECK_EQUAL(retriever.getDouble("aDouble"), 2.0);
  BOOST_CHECK_EQUAL(retriever.getInt("anInt"), 10);
  BOOST_CHECK_EQUAL(retriever.getBool("aBoolean"), true);
  BOOST_CHECK_EQUAL(retriever.getString("aString"), "somethingelse");
}

BOOST_AUTO_TEST_CASE(TestOptionsDefaults)
{
  bpo::options_description testOptions("Test options");
  testOptions.add_options()                                               //
    ("aFloat", bpo::value<float>()->default_value(10.f))                  //
    ("aDouble", bpo::value<double>()->default_value(20.))                 //
    ("anInt", bpo::value<int>()->default_value(1))                        //
    ("aBoolean", bpo::value<bool>()->zero_tokens()->default_value(false)) //
    ("aString", bpo::value<std::string>()->default_value("something"));

  FairMQProgOptions* options = new FairMQProgOptions();
  options->AddToCmdLineOptions(testOptions);
  options->ParseAll({ "cmd" }, false);
  FairOptionsRetriever retriever(options);
  BOOST_CHECK_EQUAL(retriever.getFloat("aFloat"), 10.f);
  BOOST_CHECK_EQUAL(retriever.getDouble("aDouble"), 20.);
  BOOST_CHECK_EQUAL(retriever.getInt("anInt"), 1);
  BOOST_CHECK_EQUAL(retriever.getBool("aBoolean"), false);
  BOOST_CHECK_EQUAL(retriever.getString("aString"), "something");
}
