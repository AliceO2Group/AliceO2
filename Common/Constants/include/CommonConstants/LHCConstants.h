// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file LHCConstants.h
/// \brief Header to collect LHC related constants
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_LHCCONSTANTS_H_
#define ALICEO2_LHCCONSTANTS_H_

namespace o2
{
namespace constants
{
namespace lhc
{
// LHC Beam1 and Beam2 definitions
enum BeamDirection : int { BeamClockWise, BeamAntiClockWise, NBeamDirections };
}
}
}

#endif
