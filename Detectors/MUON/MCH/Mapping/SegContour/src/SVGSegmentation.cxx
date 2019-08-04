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

#include "boost/program_options.hpp"
#include "MCHMappingInterface/CathodeSegmentation.h"
#include "MCHMappingSegContour/CathodeSegmentationContours.h"
#include "MCHMappingSegContour/CathodeSegmentationSVGWriter.h"
#include "MCHContour/SVGWriter.h"
#include <fstream>
#include <iostream>

using namespace o2::mch::mapping;

namespace po = boost::program_options;

std::pair<double, double> parsePoint(std::string ps)
{
  int ix = ps.find_first_of(' ');

  auto first = ps.substr(0, ix);
  auto second = ps.substr(ix + 1, ps.size() - ix - 1);
  return std::make_pair(std::stod(first), std::stod(second));
}

int main(int argc, char* argv[])
{

  std::string prefix;
  std::vector<int> detElemIds;
  using Point = std::pair<double, double>;
  std::vector<std::string> pointStrings;
  std::vector<Point> points;
  po::variables_map vm;
  po::options_description generic("Generic options");

  generic.add_options()("help", "produce help message")("hidepads", "hide pad outlines")(
    "hidedualsampas", "hide dualsampa outlines")("hidedes", "hide detection element outline")(
    "hidepadchannels", "hide pad channel numbering")("de", po::value<std::vector<int>>(&detElemIds),
                                                     "which detection element to consider")(
    "prefix", po::value<std::string>(&prefix)->default_value("seg"), "prefix used for outfile filename(s)")(
    "point", po::value<std::vector<std::string>>(&pointStrings), "points to show")("all", "use all detection elements");

  po::options_description cmdline;
  cmdline.add(generic);

  po::store(po::command_line_parser(argc, argv).options(cmdline).run(), vm);
  po::notify(vm);

  if (vm.count("help")) {
    std::cout << generic << "\n";
    return 2;
  }

  if (vm.count("de") && vm.count("all")) {
    std::cout << "--all and --de options are mutually exclusive. --all will be used\n";
    detElemIds.clear();
  }

  if (vm.count("all")) {
    o2::mch::mapping::forOneDetectionElementOfEachSegmentationType(
      [&detElemIds](int detElemId) { detElemIds.push_back(detElemId); });
  }

  if (detElemIds.empty()) {
    std::cout << "Must give at least one detection element id to work with\n";
    std::cout << generic << "\n";
    return 3;
  }

  for (auto ps : pointStrings) {
    points.push_back(parsePoint(ps));
  }

  for (auto& detElemId : detElemIds) {
    for (auto isBendingPlane : {true, false}) {
      std::ofstream out(vm["prefix"].as<std::string>() + "-" + std::to_string(detElemId) + "-" +
                        (isBendingPlane ? "B" : "NB") + ".html");
      CathodeSegmentation seg{detElemId, isBendingPlane};
      o2::mch::contour::SVGWriter w(getBBox(seg));
      w.addStyle(svgCathodeSegmentationDefaultStyle());
      svgCathodeSegmentation(seg, w, vm.count("hidedes") == 0, vm.count("hidedualsampas") == 0, vm.count("hidepads") == 0,
                             vm.count("hidepadchannels") == 0);
      if (!points.empty()) {
        w.svgGroupStart("testPoints");
        w.points(points, 0.2);
        w.svgGroupEnd();
      }
      w.writeHTML(out);
    }
  }

  return 0;
}
