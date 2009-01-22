//***************************************************************************
// This file is property of and copyright by the ALICE HLT Project          * 
// ALICE Experiment at CERN, All rights reserved.                           *
//                                                                          *
// Primary Authors: Sergey Gorbunov <sergey.gorbunov@kip.uni-heidelberg.de> *
//                  Ivan Kisel <kisel@kip.uni-heidelberg.de>                *
//                  for The ALICE HLT Project.                              *
//                                                                          *
// Permission to use, copy, modify and distribute this software and its     *
// documentation strictly for non-commercial purposes is hereby granted     *
// without fee, provided that the above copyright notice appears in all     *
// copies and that both the copyright notice and this permission notice     *
// appear in the supporting documentation. The authors make no claims       *
// about the suitability of this software for any purpose. It is            *
// provided "as is" without express or implied warranty.                    *
//***************************************************************************

#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCAHit.h"
#include "AliHLTTPCCARow.h"


GPUd() void AliHLTTPCCAHitAreaInit( AliHLTTPCCAHitArea &a,  AliHLTTPCCAGrid &grid, UShort_t *content, UInt_t hitoffset, Float_t y, Float_t z, Float_t dy, Float_t dz )
{ 
  // initialisation

  a.HitOffset() = hitoffset;
  a.Y() = y;
  a.Z() = z;
  a.MinZ() = z-dz; 
  a.MaxZ() = z+dz;
  a.MinY() = y-dy;
  a.MaxY() = y+dy;
  UInt_t bYmin, bZmin, bYmax;  
  grid.GetBin(a.MinY(), a.MinZ(), bYmin, bZmin);
  grid.GetBin(a.MaxY(), a.MaxZ(), bYmax, a.BZmax());  
  a.BDY() = bYmax - bYmin + 1;
  a.Ny() = grid.Ny();
  a.IndYmin() = bZmin*a.Ny() + bYmin;
  a.Iz() = bZmin;
  a.HitYfst() = content[a.IndYmin()];
  a.HitYlst() = content[a.IndYmin() + a.BDY()];    
  a.Ih() = a.HitYfst(); 
}


GPUd() void AliHLTTPCCAHitArea::Init( AliHLTTPCCAGrid &grid, UShort_t *content, UInt_t hitoffset, Float_t y, Float_t z, Float_t dy, Float_t dz )
{ 
  //initialisation
  AliHLTTPCCAHitAreaInit(*this, grid, content, hitoffset, y, z, dy, dz);
}

GPUd() Int_t AliHLTTPCCAHitArea::GetNext(AliHLTTPCCATracker &tracker, AliHLTTPCCARow &row, UShort_t *content, AliHLTTPCCAHit &h)
{    
  // get next hit index
  Float_t y0 = row.Grid().YMin();
  Float_t z0 = row.Grid().ZMin();
  Float_t stepY = row.HstepY();
  Float_t stepZ = row.HstepZ();
  uint4* tmpint4 = tracker.RowData() + row.FullOffset();
  ushort2 *hits = reinterpret_cast<ushort2*>(tmpint4);

  Int_t ret = -1;
  do{
    while( fIh>=fHitYlst ){
      if( fIz>=fBZmax ) return -1;
      fIz++;
      fIndYmin += fNy;
      fHitYfst = content[fIndYmin];
      fHitYlst = content[fIndYmin + fBDY];
      fIh = fHitYfst;
    }

    {
      ushort2 hh = hits[fIh];
      h.Y() = y0 + hh.x*stepY;
      h.Z() = z0 + hh.y*stepZ;
    }
    //h = tracker.Hits()[ fHitOffset + fIh ];    
    
    if( h.fZ>fMaxZ || h.fZ<fMinZ || h.fY<fMinY || h.fY>fMaxY ){
      fIh++;
      continue;
    }
    ret = fIh;
    fIh++;
    break;
  } while(1);
  return ret; 
}



GPUd() Int_t AliHLTTPCCAHitArea::GetBest(AliHLTTPCCATracker &tracker, AliHLTTPCCARow &row, UShort_t *content, AliHLTTPCCAHit &h)
{
  // get closest hit in the area
  Int_t best = -1;
  Float_t ds = 1.e10;
  do{
    AliHLTTPCCAHit hh;
    Int_t ih=GetNext( tracker, row, content, hh ); 
    if( ih<0 ) break;
    Float_t dy = hh.fY - fY;
    Float_t dz = hh.fZ - fZ;
    Float_t dds = dy*dy+dz*dz;
    if( dds<ds ){
      ds = dds;
      best = ih;
      h = hh;
    }
  }while(1);

  return best;
}

