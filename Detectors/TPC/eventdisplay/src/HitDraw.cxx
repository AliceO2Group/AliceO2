/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *         GNU Lesser General Public Licence version 3 (LGPL) version 3,        *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/*
 * FairMCPointsDraw.cpp
 *
 *  Created on: Apr 17, 2009
 *      Author: stockman
 */

#include "TPCEventdisplay/HitDraw.h"

#include "TPCSimulation/Point.h"                // for FairMCPoint

#include "TVector3.h"                   // for TVector3

class TObject;

using namespace o2::TPC;

HitDraw::HitDraw()
{
  // TODO Auto-generated constructor stub

}

HitDraw::~HitDraw()
{
  // TODO Auto-generated destructor stub
}

TVector3 HitDraw::GetVector(TObject* obj)
{
  Point* p = static_cast<Point*>(obj);
  return TVector3(p->GetX(), p->GetY(), p->GetZ());
}


ClassImp(HitDraw)
