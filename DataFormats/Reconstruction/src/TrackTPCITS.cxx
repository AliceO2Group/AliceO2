// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "ReconstructionDataFormats/TrackTPCITS.h"

using namespace o2::dataformats;

//__________________________________________________
void TrackTPCITS::print() const
{
  printf("TPCref: %6d ITSref: %6d\nChi2Refit: %6.2f Chi2Matc: %6.2f Time: %10.4f+-%10.4f mus\n",
         mRefTPC.getIndex(), mRefITS.getIndex(), getChi2Refit(), getChi2Match(),
         mTimeMUS.getTimeStamp(), mTimeMUS.getTimeStampError());
  printf("Inner param: ");
  o2::track::TrackParCov::print();
  printf("Outer param: ");
  mParamOut.print();
  printf("LTOF: ");
  mLTOut.print();
}
