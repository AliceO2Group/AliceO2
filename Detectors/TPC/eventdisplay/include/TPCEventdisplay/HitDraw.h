/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairMCPointsDraw.h
 *
 *  Created on: Apr 17, 2009
 *      Author: stockman
 */

#ifndef FAIRMCPOINTDRAW_H_
#define FAIRMCPOINTDRAW_H_

#include "FairPointSetDraw.h"           // for FairPointSetDraw

#include "Rtypes.h"                     // for HitDraw::Class, etc

class TObject;
class TVector3;

namespace o2 {
namespace TPC {

class HitDraw: public FairPointSetDraw
{
  public:
    HitDraw();
    HitDraw(const char* name, Color_t color ,Style_t mstyle, Int_t iVerbose = 1):FairPointSetDraw(name, color, mstyle, iVerbose) {};
    virtual ~HitDraw();

  protected:
    TVector3 GetVector(TObject* obj);

    ClassDef(HitDraw,0);
};

};
};
#endif /* FAIRMCPOINTDRAW_H_ */
