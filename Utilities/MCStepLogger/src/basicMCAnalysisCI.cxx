// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/*
 * few basic checks for analysis BasicMCAnalysisTest
 * can be used for CI in the future as soon as events are fully benchmarked
 *
 * This test requires a JSON file conatining the reference values to be tested
 * against. It has the following from
 * -----refValues.json---------
 * {
 *   "analysisName": "<analysisName>",
 *   "nSteps": [absoluteNumberOfSteps, relativeTolerance],
 *   "nTracks": [absoluteNumberOfTracks, relativeTolerance]
 * }
 * ----------------------------
 *
 * Basic usage
 * $> runTestAnalysisBasic -- <path/to/Analysis.root> <reference.json>
 */

#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK
#define BOOST_TEST_MODULE BasicMCAnalysisTest
#include <boost/test/unit_test.hpp>
#include <boost/test/results_collector.hpp>
#include <boost/test/floating_point_comparison.hpp>

// to parse the reference JSON files
#include "rapidjson/document.h"
#include "rapidjson/filereadstream.h"

#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <cstdio> // to read JSON file from disk

#include "MCStepLogger/MCAnalysisFileWrapper.h"

using namespace o2::mcstepanalysis;
namespace utf = boost::unit_test;
namespace tt = boost::test_tools;

namespace rj = rapidjson;

float integral(const TH1& histogram)
{
  return histogram.Integral();
}

// checking meta info and whether there is a value container the analysis can be conpared with
BOOST_AUTO_TEST_CASE(meta_mcbasicanalysis_test)
{
  // a path to a ROOT file containing the analysis histograms and a JSON file containing reference values, hence 3 arguments
  BOOST_REQUIRE(boost::unit_test::framework::master_test_suite().argc == 3);
  // read the file to a file wrapper
  MCAnalysisFileWrapper fileWrapper;
  BOOST_REQUIRE(fileWrapper.read(boost::unit_test::framework::master_test_suite().argv[1]));
  // read the reference JSON file
  rj::Document doc;
  std::FILE* referenceFile = std::fopen(boost::unit_test::framework::master_test_suite().argv[2], "r");
  // make sure a file was read
  BOOST_REQUIRE(referenceFile != nullptr);
  char readBuffer[65536];
  rj::FileReadStream is(referenceFile, readBuffer, sizeof(readBuffer));
  // read the JSON assuming that it's same
  doc.ParseStream(is);
  //std::cerr << doc.Size() << std::endl;
  // meta info from step logging and analysis run
  MCAnalysisMetaInfo& anaMetaInfo = fileWrapper.getAnalysisMetaInfo();
  // check only for the right analysis name, considering that the user knows which histograms to expect from this analysis
  BOOST_REQUIRE(doc.HasMember("analysisName"));
  BOOST_REQUIRE(anaMetaInfo.analysisName.compare(doc["analysisName"].GetString()) == 0);

  // require histograms, so fail if not there
  BOOST_REQUIRE(fileWrapper.hasHistogram("nSteps"));
  BOOST_REQUIRE(fileWrapper.hasHistogram("nTracks"));

  // have checked that histograms are there, so get them
  TH1& histNSteps = fileWrapper.getHistogram("nSteps");
  TH1& histNTracks = fileWrapper.getHistogram("nTracks");

  // again assuming the user passed JSON containing what is needed
  const rj::Value& nSteps = doc["nSteps"];
  const rj::Value& nTracks = doc["nTracks"];
  // ...and perform tests, assuming the JSON values are arrays
  BOOST_TEST(nSteps[0].GetFloat() == float(histNSteps.GetEntries()), tt::tolerance(nSteps[1].GetFloat()));
  BOOST_TEST(nTracks[0].GetFloat() == float(histNTracks.GetEntries()), tt::tolerance(nTracks[1].GetFloat()));
}
