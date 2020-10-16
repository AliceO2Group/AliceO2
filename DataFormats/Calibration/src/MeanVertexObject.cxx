// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#include "DataFormatsCalibration/MeanVertexObject.h"

namespace o2
{
namespace dataformats
{
  
MeanVertexObject::MeanVertexObject(const MeanVertexObject& other) {

  // copy constructor

  for (int i = 0; i < 3; i++){
    mPos[i] = other.mPos[i];
    mSigma[i] = other.mSigma[i];
  }    
  
}
  
//_____________________________________________
  MeanVertexObject::MeanVertexObject(const MeanVertexObject&& other) {

  // move constructor

  for (int i = 0; i < 3; i++){
    mPos[i] = other.mPos[i];
    mSigma[i] = other.mSigma[i];
  }    
  
}

//_____________________________________________
  MeanVertexObject& MeanVertexObject::operator = (const MeanVertexObject& other) {

    // assignment
    
  for (int i = 0; i < 3; i++){
    mPos[i] = other.mPos[i];
    mSigma[i] = other.mSigma[i];
  }
  return *this;  
}

//_____________________________________________
  MeanVertexObject& MeanVertexObject::operator = (const MeanVertexObject&& other) {

    // move assignment
    
  for (int i = 0; i < 3; i++){
    mPos[i] = other.mPos[i];
    mSigma[i] = other.mSigma[i];
  }
  return *this;  
}
  
} // dataformats
} // o2
