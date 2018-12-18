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

#ifndef O2_MCH_MAPPING_CATHODESEGMENTATIONSVGWRITER_H
#define O2_MCH_MAPPING_CATHODESEGMENTATIONSVGWRITER_H

#include <string>
#include "MCHContour/SVGWriter.h"

namespace o2
{
namespace mch
{
namespace mapping
{

class CathodeSegmentation;

std::string svgCathodeSegmentationDefaultStyle();

void svgCathodeSegmentation(const CathodeSegmentation& seg, o2::mch::contour::SVGWriter& writer, bool showdes, bool showdualsampas,
                            bool showpads, bool showpadchannels);
} // namespace mapping
} // namespace mch
} // namespace o2

#endif
