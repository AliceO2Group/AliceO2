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
#define BOOST_TEST_MODULE Test Framework DatainputDirector
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <fstream>
#include <boost/test/unit_test.hpp>

#include "Headers/DataHeader.h"
#include "../src/DataInputDirector.h"

BOOST_AUTO_TEST_CASE(TestDatainputDirector)
{
  using namespace o2::header;
  using namespace o2::framework;

  // test json file reader
  std::string jsonFile("testO2config.json");
  std::ofstream jf(jsonFile, std::ofstream::out);
  jf << R"({)" << std::endl;
  jf << R"(  "InputDirector": {)" << std::endl;
  jf << R"(    "resfiles": [)" << std::endl;
  jf << R"(      "Aresults_0.root",)" << std::endl;
  jf << R"(      "Aresults_1.root",)" << std::endl;
  jf << R"(      "Bresults_0.root",)" << std::endl;
  jf << R"(      "Bresults_1.root")" << std::endl;
  jf << R"(    ],)" << std::endl;
  jf << R"delimiter(    "fileregex": "(Ares)(.*)",)delimiter" << std::endl;
  jf << R"(    "InputDescriptors": [)" << std::endl;
  jf << R"(      {)" << std::endl;
  jf << R"(        "table": "AOD/UNO/0",)" << std::endl;
  jf << R"(        "treename": "uno")" << std::endl;
  jf << R"(      },)" << std::endl;
  jf << R"(      {)" << std::endl;
  jf << R"(        "table": "AOD/DUE/0",)" << std::endl;
  jf << R"(        "treename": "due",)" << std::endl;
  jf << R"delimiter(        "fileregex": "(Bres)(.*)")delimiter" << std::endl;
  jf << R"(      })" << std::endl;
  jf << R"(    ])" << std::endl;
  jf << R"(  })" << std::endl;
  jf << R"(})" << std::endl;
  jf.close();

  DataInputDirector didir1;
  BOOST_CHECK(didir1.readJson(jsonFile));
  didir1.printOut();
  printf("\n\n");

  BOOST_CHECK_EQUAL(didir1.getNumberInputDescriptors(), 2);

  auto dh = DataHeader(DataDescription{"DUE"},
                       DataOrigin{"AOD"},
                       DataHeader::SubSpecificationType{0});
  //auto [file1, directory1] = didir1.getFileFolder(dh, 1, 0);
  //BOOST_CHECK_EQUAL(file1->GetName(), "Bresults_1.root");

  auto didesc = didir1.getDataInputDescriptor(dh);
  BOOST_CHECK(didesc);
  BOOST_CHECK_EQUAL(didesc->getNumberInputfiles(), 2);

  // test initialization with "std::vector<std::string> inputFiles"
  // in this case "resfile" of the InputDataDirector in the json file must be
  // empty, otherwise files specified in the json file will be added to the
  // list of input files
  jf.open("testO2config.json", std::ofstream::out);
  jf << R"({)" << std::endl;
  jf << R"(  "InputDirector": {)" << std::endl;
  jf << R"delimiter(    "fileregex": "(Ares)(.*)",)delimiter" << std::endl;
  jf << R"(    "InputDescriptors": [)" << std::endl;
  jf << R"(      {)" << std::endl;
  jf << R"(        "table": "AOD/UNO/0",)" << std::endl;
  jf << R"(        "treename": "uno")" << std::endl;
  jf << R"(      },)" << std::endl;
  jf << R"(      {)" << std::endl;
  jf << R"(        "table": "AOD/DUE/0",)" << std::endl;
  jf << R"(        "treename": "due",)" << std::endl;
  jf << R"delimiter(        "fileregex": "(Bres)(.*)")delimiter" << std::endl;
  jf << R"(      })" << std::endl;
  jf << R"(    ])" << std::endl;
  jf << R"(  })" << std::endl;
  jf << R"(})" << std::endl;
  jf.close();

  std::vector<std::string> inputFiles = {"Aresults_0.root",
                                         "Aresults_1.root",
                                         "Bresults_0.root",
                                         "Aresults_2.root",
                                         "Bresults_1.root",
                                         "Bresults_2.root"};
  DataInputDirector didir2(inputFiles);
  didir2.printOut();
  printf("\n\n");
  BOOST_CHECK(didir2.readJson(jsonFile));

  //auto [file2, directory2] = didir2.getFileFolder(dh, 1, 0);
  //BOOST_CHECK_EQUAL(file2->GetName(), "Bresults_1.root");

  didesc = didir2.getDataInputDescriptor(dh);
  BOOST_CHECK(didesc);
  BOOST_CHECK_EQUAL(didesc->getNumberInputfiles(), 3);
}
