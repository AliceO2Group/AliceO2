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
/// @author  Laurent Aphecetche

#define BOOST_TEST_DYN_LINK

#include <boost/test/unit_test.hpp>

#include <boost/format.hpp>
#include "MCHMappingInterface/CathodeSegmentation.h"
#include "MCHMappingSegContour/CathodeSegmentationContours.h"
#include "MCHMappingSegContour/CathodeSegmentationSVGWriter.h"
#include "MCHContour/SVGWriter.h"
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/monomorphic/generators/xrange.hpp>
#include <boost/test/data/test_case.hpp>
#include <fstream>
#include <iostream>
#include <rapidjson/filereadstream.h>
#include <rapidjson/document.h>

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
using namespace o2::mch::mapping;
namespace bdata = boost::unit_test::data;

using Point = std::pair<double, double>;

BOOST_AUTO_TEST_SUITE(o2_mch_mapping)
BOOST_AUTO_TEST_SUITE(cathode_segmentation_long)

void dumpToFile(std::string fileName, const CathodeSegmentation& seg, const std::vector<Point>& points)
{
  std::ofstream out(fileName);
  o2::mch::contour::SVGWriter w(getBBox(seg));

  w.addStyle(svgCathodeSegmentationDefaultStyle());

  svgCathodeSegmentation(seg, w, true, true, true, false);

  w.svgGroupStart("testpoints");
  w.points(points, 0.1);
  w.svgGroupEnd();
  w.writeHTML(out);
}

/// Check that for all points within the segmentation contour
/// the findPadByPosition actually returns a valid pad
std::vector<Point> checkGaps(const CathodeSegmentation& seg, double xstep = 1.0, double ystep = 1.0)
{
  std::vector<Point> gaps;
  auto bbox = o2::mch::mapping::getBBox(seg);
  auto env = o2::mch::mapping::getEnvelop(seg);

  if (env.size() != 1) {
    throw std::runtime_error("assumption env contour = one polygon is not verified");
  }

  for (double x = bbox.xmin() - xstep; x <= bbox.xmax() + xstep; x += xstep) {
    for (double y = bbox.ymin() - ystep; y <= bbox.ymax() + ystep; y += ystep) {
      double distanceToEnveloppe =
        std::sqrt(o2::mch::contour::squaredDistancePointToPolygon(o2::mch::contour::Vertex<double>{x, y}, env[0]));
      bool withinEnveloppe = env.contains(x, y) && (distanceToEnveloppe > 1E-5);
      if (withinEnveloppe && !seg.isValid(seg.findPadByPosition(x, y))) {
        gaps.emplace_back(x, y);
      }
    }
  }
  return gaps;
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("long"))
BOOST_DATA_TEST_CASE(NoGapWithinPads,
                     boost::unit_test::data::make({100, 300, 500, 501, 502, 503, 504, 600, 601, 602, 700,
                                                   701, 702, 703, 704, 705, 706, 902, 903, 904, 905}) *
                       boost::unit_test::data::make({true, false}),
                     detElemId, isBendingPlane)
{
  CathodeSegmentation seg{detElemId, isBendingPlane};
  auto g = checkGaps(seg);

  if (!g.empty()) {
    dumpToFile("bug-gap-" + std::to_string(detElemId) + "-" + (isBendingPlane ? "B" : "NB") + ".html", seg, g);
  }
  BOOST_TEST(g.empty());
}

bool areEqual(double a, double b)
{
  return std::fabs(b - a) < 1E-4; // 1E-4 cm = 1 micron
}

int testOnePosition(const o2::mch::mapping::CathodeSegmentation& seg, rapidjson::Value& tp)
{
  double x = tp["x"].GetDouble();
  double y = tp["y"].GetDouble();
  int catPadIndex = seg.findPadByPosition(x, y);

  bool isOutside{false};

  if (tp.HasMember("isoutside")) {
    isOutside = (tp["isoutside"].GetString() == std::string("true"));
  }

  if (seg.isValid(catPadIndex) && isOutside) {
    std::cerr << "found a pad where I was not expecting one" << std::endl;
    return 1;
  }
  if (!seg.isValid(catPadIndex) && !isOutside) {
    std::cerr << "did not find a pad where I was expecting one" << std::endl;
    return 1;
  }

  if (seg.isValid(catPadIndex)) {
    double px = seg.padPositionX(catPadIndex);
    double py = seg.padPositionY(catPadIndex);

    double ex = tp["px"].GetDouble();
    double ey = tp["py"].GetDouble();
    if (!areEqual(px, ex) || !areEqual(py, ey)) {
      std::cout << "got different positions here : got px,py=" << px << "," << py << " vs expected x,y=" << ex << ","
                << ey << std::endl;
    }
  }
  return 0;
}

struct hasTestPosFile {

  boost::test_tools::assertion_result operator()(boost::unit_test_framework::test_unit_id)
  {
    path = getPath();
    return !path.empty();
  }

  std::string getPath()
  {
    auto& ts = boost::unit_test::framework::master_test_suite();
    auto n = ts.argc;
    if (n >= 2) {
      std::string opt{"--testpos"};
      for (auto i = 0; i < n - 1; i++) {
        if (opt == ts.argv[i]) {
          path = ts.argv[i + 1];
          return path;
        }
      }
    }
    return "";
  }

  std::string path;
};

// BOOST_TEST_DECORATOR( * boost::unit_test::precondition(hasTestPosFile{}) *boost::unit_test::label("long"))
// not using precondition as it is (wrongly ?) reported as an error in the run summary ?
BOOST_TEST_DECORATOR(*boost::unit_test::label("long"))
BOOST_AUTO_TEST_CASE(TestPositions)
{
  std::string filepath{hasTestPosFile{}.getPath()};

  if (filepath.empty()) {
    BOOST_TEST(true, "skipping test as to --testpos given");
    return;
  }
  InputWrapper data(filepath.c_str());

  rapidjson::Value& test_positions = data.document()["testpositions"];

  BOOST_TEST(test_positions.Size() > 0);

  int notok{0};

  for (auto& tp : test_positions.GetArray()) {
    int detElemId = tp["de"].GetInt();
    bool isBendingPlane = (tp["bending"].GetString() == std::string("true"));
    CathodeSegmentation seg(detElemId, isBendingPlane);
    notok += testOnePosition(seg, tp);
  }
  BOOST_TEST(notok == 0);
}

BOOST_TEST_DECORATOR(*boost::unit_test::label("long"))
BOOST_AUTO_TEST_CASE(TestCathodeForEachPadAndPadIndexRange)
{
  forOneDetectionElementOfEachSegmentationType([](int detElemId) {
    for (auto plane : { true, false }) {
      int n = 0;
      int pmin = std::numeric_limits<int>::max();
      int pmax = 0;
      CathodeSegmentation catseg{ detElemId, plane };
      catseg.forEachPad([&n,&pmin,&pmax](int dePadIndex) {
        n++;
        pmin = std::min(pmin,dePadIndex);
        pmax = std::max(pmax,dePadIndex);
      });
      BOOST_CHECK_EQUAL(n, catseg.nofPads());
      BOOST_CHECK_EQUAL(pmin,0);
      BOOST_CHECK_EQUAL(pmax,catseg.nofPads()-1);
    } });
}

BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
