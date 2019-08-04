// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Diego Stocco

#define BOOST_TEST_MODULE Test MID geometry transformer
#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

// #include "boost/format.hpp"
// #include <boost/test/data/monomorphic.hpp>
// #include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iostream>
#include <rapidjson/filereadstream.h>
#include <rapidjson/document.h>
#include "MIDBase/GeometryTransformer.h"

template <typename StreamType>
class InputDocument
{
 public:
  InputDocument(const char* filename)
    : mFile(fopen(filename, "r")),
      mReadBuffer(new char[65536]),
      mStream(mFile, mReadBuffer, sizeof(mReadBuffer)),
      mDocument()
  {
    mDocument.ParseStream(mStream);
  }

  rapidjson::Document& document() { return mDocument; }

  virtual ~InputDocument()
  {
    delete[] mReadBuffer;
    fclose(mFile);
  }

 private:
  FILE* mFile;
  char* mReadBuffer;
  StreamType mStream;
  rapidjson::Document mDocument;
};

using InputWrapper = InputDocument<rapidjson::FileReadStream>;
namespace bdata = boost::unit_test::data;

BOOST_AUTO_TEST_SUITE(o2_mid_simulation)

bool areEqual(double a, double b)
{
  return std::fabs(b - a) < 1E-4; // 1E-4 cm = 1 micron
}

bool areEqual(std::array<double, 3>& p1, std::array<double, 3>& p2)
{
  for (int idim = 0; idim < 3; ++idim) {
    if (!areEqual(p1[idim], p2[idim])) {
      return false;
    }
  }
  return true;
}

int testOnePosition(const o2::mid::GeometryTransformer& geoTrans, rapidjson::Value& tp)
{
  auto deId = tp["deId"].GetInt();
  auto localPoint = tp["local"].GetArray();
  auto inGlobalPoint = tp["global"].GetArray();
  std::array<double, 3> inGlob = {inGlobalPoint[0].GetDouble(), inGlobalPoint[1].GetDouble(), inGlobalPoint[2].GetDouble()};
  auto outGlobalPoint = geoTrans.localToGlobal(deId, localPoint[0].GetDouble(), localPoint[1].GetDouble());
  std::array<double, 3> outGlob = {outGlobalPoint.x(), outGlobalPoint.y(), outGlobalPoint.z()};
  if (!areEqual(inGlob, outGlob)) {
    std::cout << "got different positions for deId " << deId << " : got (" << inGlob[0] << ", " << inGlob[1] << ", " << inGlob[2] << ")  expected (" << outGlob[0] << ", " << outGlob[1] << ", " << outGlob[2] << ")\n";
    return 1;
  }
  return 0;
}

std::string getTestPosFilename()
{
  std::string path = "test_random_pos.json";
  auto& ts = boost::unit_test::framework::master_test_suite();
  auto nargs = ts.argc;
  if (nargs >= 2) {
    std::string opt{"--testpos"};
    for (auto iarg = 0; iarg < nargs - 1; iarg++) {
      if (opt == ts.argv[iarg]) {
        path = ts.argv[iarg + 1];
        break;
      }
    }
  }
  return path;
}

bool fileExists(const char* filename)
{
  std::ifstream file(filename);
  return file.good();
}

BOOST_AUTO_TEST_CASE(TestPositions)
{
  std::string filepath = getTestPosFilename();

  if (!fileExists(filepath.c_str())) {
    BOOST_TEST(false, "No test file found. Do it with: test_MIDpositions -- --testpos <testFilename>");
    return;
  }
  InputWrapper data(filepath.c_str());

  rapidjson::Value& test_positions = data.document()["testpositions"];

  BOOST_TEST(test_positions.Size() > 0);

  int notok{0};

  o2::mid::GeometryTransformer geoTrans = o2::mid::createDefaultTransformer();

  for (auto& tp : test_positions.GetArray()) {
    notok += testOnePosition(geoTrans, tp);
  }
  BOOST_TEST(notok == 0);
}

BOOST_AUTO_TEST_SUITE_END()
