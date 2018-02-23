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

#ifndef O2_MCH_MAPPING_SEGMENTATIONCONTOURS_H
#define O2_MCH_MAPPING_SEGMENTATIONCONTOURS_H

#include <vector>
#include "MCHContour/Contour.h"
#include "MCHContour/Polygon.h"
#include "MCHContour/BBox.h"
#include "MCHMappingInterface/Segmentation.h"

namespace o2
{
namespace mch
{
namespace mapping
{

std::vector<std::vector<o2::mch::contour::Polygon<double>>> getPadPolygons(const Segmentation& seg);

std::vector<std::vector<int>> getPadChannels(const Segmentation& seg);

std::vector<o2::mch::contour::Contour<double>> getDualSampaContours(const Segmentation& seg);

o2::mch::contour::Contour<double> getEnvelop(const Segmentation& seg);

o2::mch::contour::BBox<double> getBBox(const Segmentation& seg);
} // namespace mapping
} // namespace mch
} // namespace o2

#endif
