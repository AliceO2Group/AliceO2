/**************************************************************************
 * Copyright(c) 1998-2007, ALICE Experiment at CERN, All rights reserved. *
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
//     MC Vertex class
//     Class to be used for Kinematics MC Data
//     andreas.morsch@cern.ch
//-------------------------------------------------------------------------

#include "AliMCVertex.h"
void AliMCVertex::Print(Option_t* /*option*/) const
{
    printf("MC Primary Vertex Position x = %13.3f, y = %13.3f, z = %13.3f \n",
	   fPosition[0], fPosition[1], fPosition[2]);
    
}

ClassImp(AliMCVertex)
