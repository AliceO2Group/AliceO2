// **************************************************************************
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
//                                                                          *
//***************************************************************************

#include "AliHLTTPCCAHitArea.h"
#include "AliHLTTPCCATracker.h"
#include "AliHLTTPCCAGrid.h"
#include "AliHLTTPCCAHit.h"
class AliHLTTPCCARow;


GPUdi() void AliHLTTPCCAHitArea::Init( const AliHLTTPCCARow &row, const AliHLTTPCCASliceData &slice, float y, float z,
                                      float dy, float dz )
{
  //initialisation
  const AliHLTTPCCAGrid &grid = row.Grid();
  fHitOffset = row.HitNumberOffset();
  fY = y;
  fZ = z;
  fMinZ = z - dz;
  fMaxZ = z + dz;
  fMinY = y - dy;
  fMaxY = y + dy;
  int bYmin, bZmin, bYmax; // boundary bin indexes
  grid.GetBin( fMinY, fMinZ, &bYmin, &bZmin );
  grid.GetBin( fMaxY, fMaxZ, &bYmax, &fBZmax );
  fBDY = bYmax - bYmin + 1; // bin index span in y direction
  fNy = grid.Ny();
  fIndYmin = bZmin * fNy + bYmin; // same as grid.GetBin(fMinY, fMinZ), i.e. the smallest bin index of interest
  // fIndYmin + fBDY then is the largest bin index of interest with the same Z
  fIz = bZmin;

  // for given fIz (which is min atm.) get
#ifdef HLTCA_GPU_TEXTURE_FETCHa
  fHitYfst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(unsigned short) + fIndYmin);
  fHitYlst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(unsigned short) + fIndYmin + fBDY);
#else
  fHitYfst = slice.FirstHitInBin( row, fIndYmin ); // first and
  fHitYlst = slice.FirstHitInBin( row, fIndYmin + fBDY ); // last hit index in the bin
#endif
  fIh = fHitYfst;
}

GPUdi() int AliHLTTPCCAHitArea::GetNext( const AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row,
                                        const AliHLTTPCCASliceData &slice, AliHLTTPCCAHit *h )
{
  // get next hit index

  // min coordinate
  const float y0 = row.Grid().YMin();
  const float z0 = row.Grid().ZMin();

  // step vector
  const float stepY = row.HstepY();
  const float stepZ = row.HstepZ();

  int ret = -1;
  do {
    while ( fIh >= fHitYlst ) {
      if ( fIz >= fBZmax ) {
        return -1;
      }
      // go to next z and start y from the min again
      ++fIz;
      fIndYmin += fNy;
#ifdef HLTCA_GPU_TEXTURE_FETCHa
	  fHitYfst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(unsigned short) + fIndYmin);
	  fHitYlst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(unsigned short) + fIndYmin + fBDY);
#else
      fHitYfst = slice.FirstHitInBin( row, fIndYmin );
      fHitYlst = slice.FirstHitInBin( row, fIndYmin + fBDY );
#endif
      fIh = fHitYfst;
    }

#ifdef HLTCA_GPU_TEXTURE_FETCHa
	ushort2 tmpval = tex1Dfetch(gAliTexRefu2, ((char*) slice.HitData(row) - slice.GPUTextureBaseConst()) / sizeof(ushort2) + fIh);;
	h->SetY( y0 + tmpval.x * stepY );
    h->SetZ( z0 + tmpval.y * stepZ );
#else
	h->SetY( y0 + tracker.HitDataY( row, fIh ) * stepY );
    h->SetZ( z0 + tracker.HitDataZ( row, fIh ) * stepZ );
#endif

    if ( 1 && ( h->Z() > fMaxZ || h->Z() < fMinZ || h->Y() < fMinY || h->Y() > fMaxY ) ) { //SG!!!
      fIh++;
      continue;
    }
    ret = fIh;
    fIh++;
    break;
  } while ( 1 );
  return ret;
}


/*
int AliHLTTPCCAHitArea::GetBest( const AliHLTTPCCATracker &tracker, const AliHLTTPCCARow &row,
    const int *content, AliHLTTPCCAHit *h)
{
  // get closest hit in the area
  int best = -1;
  float ds = 1.e10;
  do {
    AliHLTTPCCAHit hh;
    int ih = GetNext( tracker, row, content, hh );
    if ( ih < 0 ) break;
    float dy = hh.Y() - fY;
    float dz = hh.Z() - fZ;
    float dds = dy * dy + dz * dz;
    if ( dds < ds ) {
      ds = dds;
      best = ih;
      h = hh;
    }
  } while ( 1 );

  return best;
}
*/
