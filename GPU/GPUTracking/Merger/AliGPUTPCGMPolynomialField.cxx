// **************************************************************************
// This file is property of and copyright by the ALICE HLT Project          *
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Vito Nordloh <vito.nordloh@vitonordloh.de>              *
//                  Sergey Gorbunov <sergey.gorbunov@fias.uni-frankfurt.de> *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//                                                                          *
//***************************************************************************

#include "AliGPUTPCGMPolynomialField.h"


#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

void AliGPUTPCGMPolynomialField::Print() const
{
  const double kCLight = 0.000299792458;
  typedef std::numeric_limits< float > flt;
  cout<<std::scientific;
#if __cplusplus >= 201103L
  cout<<std::setprecision( flt::max_digits10+2 );
#endif
  cout<<" nominal field "<< fNominalBz <<" [kG * (2.99792458E-4 GeV/c/kG/cm)]"
      <<" == "<<fNominalBz/kCLight<<" [kG]"<<endl;
  
  cout<<" TpcBx[fkTpcM] = { ";
  for( int i=0; i<fkTpcM; i++){
    cout<<fTpcBx[i];
    if( i<fkTpcM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" TpcBy[fkTpcM] = { ";
  for( int i=0; i<fkTpcM; i++){
    cout<<fTpcBy[i];
    if( i<fkTpcM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" TpcBz[fkTpcM] = { ";
  for( int i=0; i<fkTpcM; i++){
    cout<<fTpcBz[i];
    if( i<fkTpcM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }
  
  cout<<"TRD field: \n"<<endl;
  
  cout<<" TrdBx[fkTrdM] = { ";
  for( int i=0; i<fkTrdM; i++){
    cout<<fTrdBx[i];
    if( i<fkTrdM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" TrdBy[fkTrdM] = { ";
  for( int i=0; i<fkTrdM; i++){
    cout<<fTrdBy[i];
    if( i<fkTrdM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" TrdBz[fkTrdM] = { ";
  for( int i=0; i<fkTrdM; i++){
    cout<<fTrdBz[i];
    if( i<fkTrdM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<"ITS field: \n"<<endl;
  
  cout<<" ItsBx[fkItsM] = { ";
  for( int i=0; i<fkItsM; i++){
    cout<<fItsBx[i];
    if( i<fkItsM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" ItsBy[fkItsM] = { ";
  for( int i=0; i<fkItsM; i++){
    cout<<fItsBy[i];
    if( i<fkItsM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" ItsBz[fkItsM] = { ";
  for( int i=0; i<fkItsM; i++){
    cout<<fItsBz[i];
    if( i<fkItsM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }
}

#else

void AliGPUTPCGMPolynomialField::Print() const
{
  // do nothing
}

#endif
