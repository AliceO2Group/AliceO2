// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackMFT.cxx
/// \brief Implementation of the MFT track
/// \author bogdan.vulpescu@cern.ch
/// \date Feb. 8, 2018

#include "DataFormatsMFT/TrackMFT.h"
#include "CommonConstants/MathConstants.h"
#include "DataFormatsITSMFT/Cluster.h"

ClassImp(o2::MFT::TrackMFT);

using namespace o2::MFT;
using namespace o2::ITSMFT;
using namespace o2::constants::math;
using namespace o2::track;

//_____________________________________________________________________________
