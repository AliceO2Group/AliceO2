// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "AnalysisConfigurableCuts.h"

using namespace o2::analysis;

ClassImp(SimpleInclusiveCut);

SimpleInclusiveCut::SimpleInclusiveCut() : TNamed(),
                                           mX(0),
                                           mY(0.0)
{
  //
  // default constructor
  //
}

SimpleInclusiveCut::SimpleInclusiveCut(const char* name, int _x, float _y) : TNamed(name, name),
                                                                             mX(_x),
                                                                             mY(_y)
{
  //
  // explicit constructor
  //
}

SimpleInclusiveCut& SimpleInclusiveCut::operator=(const SimpleInclusiveCut& sic)
{
  //
  // assignment operator
  //
  if (this != &sic) {
    TNamed::operator=(sic);
    mX = sic.mX;
    mY = sic.mY;
  }
  return (*this);
}
