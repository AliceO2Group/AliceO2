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

//-------------------------------------------------------------------------
//                        Implemenation Class AliESDFIT
//   This is a class that summarizes the FIT data for the ESD   
//   Origin: Alla Maevskaya Alla.Maevskaya@cern.ch 
//-------------------------------------------------------------------------



#include "AliESDFIT.h"
#include "AliLog.h"


ClassImp(AliESDFIT)

//______________________________________________________________________________
AliESDFIT::AliESDFIT() :
  TObject(),
  fFITzVertex(0)
{
  
  for(int i=0; i<3; i++) {
    fT0[i] = -99999;
    fT0best[i] = -99999;
  }
  for(int i=0; i<288; i++) {
    fFITtime[i] = -99999;
    fFITamplitude[i] = 0;
    fFITphotons[i] = 0;
  }


}
//______________________________________________________________________________
AliESDFIT::AliESDFIT(const AliESDFIT &tzero ) :
  TObject(tzero),  
  fFITzVertex(tzero.fFITzVertex)
 {
  // copy constuctor
  for(int i=0; i<3; i++) {
    fT0[i] = tzero.fT0[i];
    fT0best[i] = tzero.fT0best[i];
  }
  for(int i=0; i<288; i++) {
    fFITtime[i] = -99999;
    fFITamplitude[i] = 0;
    fFITphotons[i] = 0;
  }
 }
//______________________________________________________________________________
AliESDFIT& AliESDFIT::operator=(const AliESDFIT& tzero){
  // assigmnent operator
  if(this!=&tzero) {
    TObject::operator=(tzero);
      fFITzVertex = tzero.fFITzVertex;
     for(int i=0; i<3; i++) {
      fT0[i] = tzero.fT0[i];
      fT0best[i] = tzero.fT0best[i];
    }

   for(int i=0; i<288; i++){
      fFITtime[i] = tzero.fFITtime[i]; 
      fFITamplitude[i] = tzero.fFITamplitude[i];
      fFITphotons[i] = tzero.fFITphotons[i];

     }
   }
   
  return *this;
}
//______________________________________________________________________________
void AliESDFIT::Copy(TObject &obj) const {
  
  // this overwrites the virtual TOBject::Copy()
  // to allow run time copying without casting
  // in AliESDEvent

  if(this==&obj)return;
  AliESDFIT *robj = dynamic_cast<AliESDFIT*>(&obj);
  if(!robj)return; // not an AliESDFIT
  *robj = *this;

}


//______________________________________________________________________________
void AliESDFIT::Reset()
{
  // reset contents
  fFITzVertex = -9999;  
  for(int i=0; i<288; i++) {
    fFITtime[i] = fFITamplitude[i] =  0;
    fFITtime[i] = fFITphotons[i] =  0;
  }
  for(int i=0; i<3 ;i++) {
    fT0[i] = -9999;
    fT0best[i] = -9999;
  }
}

//______________________________________________________________________________
void AliESDFIT::Print(const Option_t *) const
{
  // does noting fornow
  AliInfo(Form(" Vertex %f  T0signal %f ns OrA %f ns OrC %f \n",fFITzVertex,  fT0[0],fT0[1],fT0[2]));

}
