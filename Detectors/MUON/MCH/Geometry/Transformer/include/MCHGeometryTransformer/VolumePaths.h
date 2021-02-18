// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// get the local-to-global transformation for a given detection element

#ifndef O2_MCH_GEOMETRY_TRANSFORMER_VOLUME_PATHS_H
#define O2_MCH_GEOMETRY_TRANSFORMER_VOLUME_PATHS_H

#include <string>

namespace o2::mch::geo
{
std::string volumePathName(int deId);
}

#endif
