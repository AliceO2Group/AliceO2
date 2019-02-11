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

/* $Id$ */
//-------------------------------------------------------------------------
//                          Class AliCollisionGeometry
//   This is a class to handle the collison geometry defined by
//   the generator
//   Author: A.Morsch Andreas.Morsch@cern.ch
//-------------------------------------------------------------------------


#include "AliCollisionGeometry.h"
ClassImp(AliCollisionGeometry)

//______________________________________________________________________
AliCollisionGeometry::AliCollisionGeometry() :
  fNHardScatters(0),
  fNProjectileParticipants(0),
  fNTargetParticipants(0),
  fNNColl(0),
  fNNwColl(0),
  fNwNColl(0),
  fNwNwColl(0),
  fProjectileSpecn(0),
  fProjectileSpecp(0),
  fTargetSpecn(0),
  fTargetSpecp(0),
  fImpactParameter(0),
  fReactionPlaneAngle(0),
  fNSD1(0),
  fNSD2(0),
  fNDD(0)
{
}

AliCollisionGeometry::AliCollisionGeometry(const AliCollisionGeometry& cgeo):
  fNHardScatters(cgeo.fNHardScatters),
  fNProjectileParticipants(cgeo.fNProjectileParticipants),
  fNTargetParticipants(cgeo.fNTargetParticipants),
  fNNColl(cgeo.fNNColl),
  fNNwColl(cgeo.fNNwColl),
  fNwNColl(cgeo.fNwNColl),
  fNwNwColl(cgeo.fNwNwColl),
  fProjectileSpecn(cgeo.fProjectileSpecn),
  fProjectileSpecp(cgeo.fProjectileSpecp),
  fTargetSpecn(cgeo.fTargetSpecn),
  fTargetSpecp(cgeo.fTargetSpecp),
  fImpactParameter(cgeo.fImpactParameter),
  fReactionPlaneAngle(cgeo.fReactionPlaneAngle),
  fNSD1(cgeo.fNSD1),
  fNSD2(cgeo.fNSD2),
  fNDD(cgeo.fNDD)
{
    // Copy constructor
}
