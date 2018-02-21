//
// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See https://alice-o2.web.cern.ch/ for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// @author  Laurent Apaduidecetche

#include "MCHMappingSegContour/SegmentationSVGWriter.h"
#include "MCHMappingInterface/Segmentation.h"
#include "MCHMappingSegContour/SegmentationContours.h"
#include "MCHContour/SVGWriter.h"
#include <ostream>

using namespace o2::mch::contour;

namespace o2 {
namespace mch {
namespace mapping {

std::string svgSegmentationDefaultStyle()
{
  return R"(
.pads {
  fill: #EEEEEE;
  stroke-width: 0.025px;
  stroke: #AAAAAA;
}
.padchannels {
  font-size: 0.4px;
  font-family: arial;
  fill: blue;
  text-anchor: middle;
}
.dualsampas {
  fill:none;
  stroke-width: 0.025px;
  stroke: #333333;
}
.detectionelements {
  fill:none;
  stroke-width:0.025px;
  stroke: #000000;
}
.testpoints {
  fill:red;
  stroke-width:0.025px;
  stroke: black;
  opacity: 0.5;
}
)";
}

void svgSegmentation(const Segmentation &seg, SVGWriter &w,
                     bool showdes, bool showdualsampas, bool showpads,
                     bool showpadchannels)
{
  std::vector<Contour<double>> dualSampaContours = getDualSampaContours(seg);
  std::vector<std::vector<Polygon<double>>> dualSampaPads = getPadPolygons(seg);
  std::vector<std::vector<int>> dualSampaPadChannels = getPadChannels(seg);

  if (dualSampaPadChannels.size() != dualSampaPads.size()) {
    throw std::runtime_error("gouze");
  }

  auto deContour = getEnvelop(seg);
  auto box = getBBox(seg);

  if (showpads) {
    w.svgGroupStart("pads");
    for (auto &dsp: dualSampaPads) {
      for (auto &p: dsp) {
        w.polygon(p);
      }
    }
    w.svgGroupEnd();
  }
 
  if (showpadchannels) {
    w.svgGroupStart("padchannels");
    for (auto i = 0; i < dualSampaPads.size(); ++i) {
      auto &dsp = dualSampaPads[i];
      auto &dspch = dualSampaPadChannels[i];
      for (auto j = 0; j < dsp.size(); j++) {
        auto bbox = getBBox(dsp[j]);
        w.text(std::to_string(dspch[j]), bbox.xcenter(),
               bbox.ymax() - 0.05 * bbox.height()); // SVG text y position is the bottom of the text
      }
    }
    w.svgGroupEnd();
  }

  if (showdualsampas) {
    w.svgGroupStart("dualsampas");
    for (auto &dsp: dualSampaContours) {
      w.contour(dsp);
    }
    w.svgGroupEnd();
  }

  if (showdes) {
    w.svgGroupStart("detectionelements");
    w.contour(deContour);
  }

}

}
}
}

