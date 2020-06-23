// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#define BOOST_TEST_MODULE Test Framework DatainputDirector
#define BOOST_TEST_MAIN
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include "Headers/DataHeader.h"
#include "Framework/DataInputDirector.h"

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

  DataInputDirector didir;
  BOOST_CHECK(didir.readJson(jsonFile));
  //didir.printOut(); printf("\n\n");

  BOOST_CHECK_EQUAL(didir.getNumberInputDescriptors(), 2);

  auto dh = DataHeader(DataDescription{"DUE"},
                       DataOrigin{"AOD"},
                       DataHeader::SubSpecificationType{0});
  BOOST_CHECK_EQUAL(didir.getInputFilename(dh, 1), "Bresults_1.root");

  auto didesc = didir.getDataInputDescriptor(dh);
  BOOST_CHECK(didesc);
  BOOST_CHECK_EQUAL(didesc->getNumberInputfiles(), 2);
}
