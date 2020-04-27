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

#ifndef O2_MCH_MAPPING_impl4_CATHODESEGMENTATIONCREATOR_H
#define O2_MCH_MAPPING_impl4_CATHODESEGMENTATIONCREATOR_H

#include "CathodeSegmentationImpl4.h"

namespace o2
{
namespace mch
{
namespace mapping
{
namespace impl4
{

using CathodeSegmentationCreator = CathodeSegmentation* (*)(bool);

void registerCathodeSegmentationCreator(int segType,
                                        CathodeSegmentationCreator func);

CathodeSegmentationCreator getCathodeSegmentationCreator(int segType);

} // namespace impl4
} // namespace mapping
} // namespace mch
} // namespace o2

#endif
