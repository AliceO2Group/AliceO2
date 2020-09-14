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
#include "Framework/Logger.h"
#include "MathUtils/Utils.h"

namespace o2
{
namespace mft
{

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

//__________________________________________________________________________
void TrackMFT::print() const
{
  /// Printing TrackMFT information
  LOG(INFO) << "TrackMFT: p =" << std::setw(5) << std::setprecision(3) << getP()
            << " Tanl = " << std::setw(5) << std::setprecision(3) << getTanl()
            << " phi = " << std::setw(5) << std::setprecision(3) << getPhi()
            << " pz = " << std::setw(5) << std::setprecision(3) << getPz()
            << " pt = " << std::setw(5) << std::setprecision(3) << getPt()
            << " charge = " << std::setw(5) << std::setprecision(3) << getCharge()
            << " chi2 = " << std::setw(5) << std::setprecision(3) << getTrackChi2() << std::endl;
}


} // namespace mft
} // namespace o2
