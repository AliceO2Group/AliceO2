// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
/// \author Nima Zardoshti <nima.zardoshti@cern.ch>, CERN

#ifndef O2_ANALYSIS_UDDERIVED_H
#define O2_ANALYSIS_UDDERIVED_H

#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{

namespace udtrack
{
DECLARE_SOA_COLUMN(Pt1, pt1, float);
DECLARE_SOA_COLUMN(Eta1, eta1, float);
DECLARE_SOA_COLUMN(Phi1, phi1, float);
DECLARE_SOA_COLUMN(TpcSignal1, tpcSignal1, float);
DECLARE_SOA_COLUMN(Pt2, pt2, float);
DECLARE_SOA_COLUMN(Eta2, eta2, float);
DECLARE_SOA_COLUMN(Phi2, phi2, float);
DECLARE_SOA_COLUMN(TpcSignal2, tpcSignal2, float);

} // namespace udtrack
DECLARE_SOA_TABLE(UDTracks, "AOD", "UDTRACK", o2::soa::Index<>,
                  udtrack::Pt1, udtrack::Eta1, udtrack::Phi1,
                  udtrack::TpcSignal1,
                  udtrack::Pt2, udtrack::Eta2, udtrack::Phi2,
                  udtrack::TpcSignal2);
using UDTrack = UDTracks::iterator;

} // namespace o2::aod

#endif // O2_ANALYSIS_UDDERIVED_H
