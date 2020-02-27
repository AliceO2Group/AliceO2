// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_MCH_MAPPING_FACTORY_CREATE_SEGMENTATION_H
#define O2_MCH_MAPPING_FACTORY_CREATE_SEGMENTATION_H

#include "MCHMappingInterface/Segmentation.h"

namespace o2::mch::mapping
{
/// segmentation is a convenience function that
/// returns a segmentation for a given detection element.
///
/// That segmentation is part of a pool of Segmentations
/// (one per DE) handled by this module.
///
/// Note that this may be not be what you want as this module
/// will always create one (but only one) Segmentation for
/// all 156 detection elements.
/// For instance, if you know you'll be dealing with only
/// one detection element, you'd be better using
/// the Segmentation ctor simply, and ensure by yourself
/// that you are only creating it once in order not to incur
/// the (high) price of the construction time of that Segmentation.
const Segmentation& segmentation(int detElemId);
} // namespace o2::mch::mapping

#endif
