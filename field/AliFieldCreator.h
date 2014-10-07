/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/

// -------------------------------------------------------------------------
// -----                    AliFieldCreator header file                  -----
// -----                Created 26/03/14  by M. Al-Turany              -----
// -------------------------------------------------------------------------


#ifndef AliFieldCreator_H
#define AliFieldCreator_H

#include "FairFieldFactory.h"

class AliFieldPar;

class FairField;

class AliFieldCreator : public FairFieldFactory 
{

 public:
  AliFieldCreator();
  virtual ~AliFieldCreator();
  virtual FairField* createFairField();
  virtual void SetParm();
  ClassDef(AliFieldCreator,1);
  
 protected:
  AliFieldPar* fFieldPar;
  
 private:
  AliFieldCreator(const AliFieldCreator&);
  AliFieldCreator& operator=(const AliFieldCreator&);

};
#endif //AliFieldCreator_H
