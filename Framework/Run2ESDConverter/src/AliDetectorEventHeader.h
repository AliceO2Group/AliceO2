#ifndef ALIDETECTOREVENTHEADER_H
#define ALIDETECTOREVENTHEADER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

//---------------------------------------------------------------------
// Event header base class for detetors 
// Stores detector specific information
// Author: andreas.morsch@cern.ch
//---------------------------------------------------------------------

#include <TNamed.h>
class AliDetectorEventHeader : public TNamed
{
 public:

  AliDetectorEventHeader(const char* name);
  AliDetectorEventHeader();
  virtual ~AliDetectorEventHeader() {}
protected:
  ClassDef(AliDetectorEventHeader,0) // Event header for detectors
};

#endif
