// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/TrackCosmics.h"

using namespace o2::dataformats;

//__________________________________________________
void TrackCosmics::print() const
{
  printf("Bottom: %s/%d Top: %s/%d Chi2Refit: %6.2f Chi2Match: %6.2f Ncl: %d Time: %10.4f+-%10.4f mus\n",
         mRefBottom.getSourceName().data(), mRefBottom.getIndex(), mRefTop.getSourceName().data(), mRefTop.getIndex(),
         getChi2Refit(), getChi2Match(), getNClusters(), mTimeMUS.getTimeStamp(), mTimeMUS.getTimeStampError());
  printf("Central param: ");
  o2::track::TrackParCov::print();
  printf("Outer param: ");
  mParamOut.print();
}
