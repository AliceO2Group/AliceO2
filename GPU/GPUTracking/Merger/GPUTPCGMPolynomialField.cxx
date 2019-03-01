// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file GPUTPCGMPolynomialField.cxx
/// \author Sergey Gorbunov, David Rohr

#include "GPUTPCGMPolynomialField.h"

#if defined(GPUCA_ALIROOT_LIB) & !defined(GPUCA_GPUCODE)

#include <iostream>
#include <iomanip>
#include <limits>

using namespace std;

void GPUTPCGMPolynomialField::Print() const
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

void GPUTPCGMPolynomialField::Print() const
{
  // do nothing
}

#endif
