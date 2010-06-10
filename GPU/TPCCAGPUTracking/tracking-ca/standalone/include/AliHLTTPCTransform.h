// @(#) $Id: AliHLTTPCTransform.h 32256 2009-05-08 08:25:50Z richterm $
// Original: AliHLTTransform.h,v 1.37 2005/06/14 10:55:21 cvetan 

//* This file is property of and copyright by the ALICE HLT Project        * 
//* ALICE Experiment at CERN, All rights reserved.                         *
//* See cxx source for full Copyright notice                               *

#ifndef ALIHLTTPCTRANSFORM_H
#define ALIHLTTPCTRANSFORM_H

#ifdef use_aliroot
  class AliRunLoader;
#endif

#include "AliHLTTPCRootTypes.h"

class AliHLTTPCTransform {

 public:
    AliHLTTPCTransform();
  enum VersionType { kVdefault=0, kVdeprecated=1, kValiroot=10, kVcosmics=100};

 private:
  static Double_t fgX[159];  //X position in local coordinates

 public:
  virtual ~AliHLTTPCTransform() {}

  static Double_t Row2X(Int_t slicerow);
  
  ClassDef(AliHLTTPCTransform,1)
};
#endif
