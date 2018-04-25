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

#include "AliHLTTPCGMPolynomialField.h"


#if !defined(HLTCA_STANDALONE) & !defined(HLTCA_GPUCODE)

#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

void AliHLTTPCGMPolynomialField::Print() const
{
  const double kCLight = 0.000299792458;
  typedef std::numeric_limits< float > flt;  
  cout<<std::scientific;
#if __cplusplus >= 201103L  
  cout<<std::setprecision( flt::max_digits10+2 );
#endif
  cout<<" nominal field "<< fNominalBz <<" [kG * (2.99792458E-4 GeV/c/kG/cm)]"
      <<" == "<<fNominalBz/kCLight<<" [kG]"<<endl;
  
  cout<<" Bx[fkM] = { ";
  for( int i=0; i<fkM; i++){
    cout<<fBx[i];
    if( i<fkM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" By[fkM] = { ";
  for( int i=0; i<fkM; i++){
    cout<<fBy[i];
    if( i<fkM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }

  cout<<" Bz[fkM] = { ";
  for( int i=0; i<fkM; i++){
    cout<<fBz[i];
    if( i<fkM-1 ) cout<<", ";
    else cout<<" };"<<endl;
  }
}

#else

void AliHLTTPCGMPolynomialField::Print() const
{
  // do nothing
}

#endif
