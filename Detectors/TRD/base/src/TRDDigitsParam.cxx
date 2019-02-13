// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/* $Id: TRDDigitsParam.cxx 34070 2009-08-04 15:34:53Z cblume $ */

///////////////////////////////////////////////////////////////////////////////
//                                                                           //
//  Class containing parameters for digits                                   //
//  2019 - Ported from AliRoot to O2 (J. Lopez)                              //
//                                                                           //
///////////////////////////////////////////////////////////////////////////////

#include "TRDBase/TRDDigitsParam.h"

using namespace o2::trd;

//_____________________________________________________________________________
TRDDigitsParam::TRDDigitsParam(const TRDDigitsParam& p)
{
  //
  // Copy constructor
  //

  mNTimeBins = p.mNTimeBins;
  mPretriggerPhase = p.mPretriggerPhase;
  mADCbaseline = p.mADCbaseline;
}

//_____________________________________________________________________________
TRDDigitsParam& TRDDigitsParam::operator=(const TRDDigitsParam& p)
{
  //
  // Assignment operator
  //

  if (this != &p) {
    ((TRDDigitsParam&)p).Copy(*this);
  }

  return *this;
}

//_____________________________________________________________________________
void TRDDigitsParam::Copy(TRDDigitsParam& p) const
{
  //
  // Copy function
  //

  TRDDigitsParam* target = dynamic_cast<TRDDigitsParam*>(&p);
  if (!target) {
    return;
  }

  target->mNTimeBins = mNTimeBins;
  target->mPretriggerPhase = mPretriggerPhase;
  target->mADCbaseline = mADCbaseline;
}
