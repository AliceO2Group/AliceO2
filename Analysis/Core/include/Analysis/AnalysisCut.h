// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
//
// Contact: iarsene@cern.ch, i.c.arsene@fys.uio.no
//
// Base class for analysis cuts
//

#ifndef AnalysisCut_H
#define AnalysisCut_H

#include <TNamed.h>

//_________________________________________________________________________
class AnalysisCut : public TNamed
{

 public:
  AnalysisCut();
  AnalysisCut(const char* name, const char* title);
  ~AnalysisCut() override;
  
  bool IsSelected(float* values) {return kTRUE;};
  //TODO: include also IsSelected() functions which handle various objects
  
 protected: 
   
  AnalysisCut(const AnalysisCut &c);
  AnalysisCut& operator= (const AnalysisCut &c);
  
  ClassDef(AnalysisCut,1);
  
};
#endif
