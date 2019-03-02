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
MEM_CLASS_PRE() class AliHLTTPCCARow;

MEM_TEMPLATE() GPUdi() void AliHLTTPCCAHitArea::Init( const MEM_TYPE( AliHLTTPCCARow) &row, GPUglobalref() const MEM_GLOBAL(AliHLTTPCCASliceData) &slice, float y, float z,
                                      float dy, float dz )
{
  //initialisation
  fHitOffset = row.HitNumberOffset();
  fY = y;
  fZ = z;
  fMinZ = z - dz;
  fMaxZ = z + dz;
  fMinY = y - dy;
  fMaxY = y + dy;
  int bYmin, bZmin, bYmax; // boundary bin indexes
  row.Grid().GetBin( fMinY, fMinZ, &bYmin, &bZmin );
  row.Grid().GetBin( fMaxY, fMaxZ, &bYmax, &fBZmax );
  fBDY = bYmax - bYmin + 1; // bin index span in y direction
  fNy = row.Grid().Ny();
  fIndYmin = bZmin * fNy + bYmin; // same as grid.GetBin(fMinY, fMinZ), i.e. the smallest bin index of interest
  // fIndYmin + fBDY then is the largest bin index of interest with the same Z
  fIz = bZmin;

  // for given fIz (which is min atm.) get
#ifdef HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
  fHitYfst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + fIndYmin);
  fHitYlst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + fIndYmin + fBDY);
#else
  fHitYfst = slice.FirstHitInBin( row, fIndYmin ); // first and
  fHitYlst = slice.FirstHitInBin( row, fIndYmin + fBDY ); // last hit index in the bin
#endif //HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
  fIh = fHitYfst;
}

MEM_TEMPLATE() GPUdi() int AliHLTTPCCAHitArea::GetNext( GPUconstant() const MEM_CONSTANT(AliHLTTPCCATracker) &tracker, const MEM_TYPE( AliHLTTPCCARow) &row,
                                        GPUglobalref() const MEM_GLOBAL(AliHLTTPCCASliceData) &slice, AliHLTTPCCAHit *h )
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
#ifdef HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
	  fHitYfst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + fIndYmin);
	  fHitYlst = tex1Dfetch(gAliTexRefu, ((char*) slice.FirstHitInBin(row) - slice.GPUTextureBaseConst()) / sizeof(calink) + fIndYmin + fBDY);
#else
      fHitYfst = slice.FirstHitInBin( row, fIndYmin );
      fHitYlst = slice.FirstHitInBin( row, fIndYmin + fBDY );
#endif
      fIh = fHitYfst;
    }

#ifdef HLTCA_GPU_TEXTURE_FETCH_NEIGHBORS
	cahit2 tmpval = tex1Dfetch(gAliTexRefu2, ((char*) slice.HitData(row) - slice.GPUTextureBaseConst()) / sizeof(cahit2) + fIh);;
	h->SetY( y0 + tmpval.x * stepY );
    h->SetZ( z0 + tmpval.y * stepZ );
#else
	h->SetY( y0 + tracker.HitDataY( row, fIh ) * stepY );
    h->SetZ( z0 + tracker.HitDataZ( row, fIh ) * stepZ );
#endif

    if ( h->Z() > fMaxZ || h->Z() < fMinZ || h->Y() < fMinY || h->Y() > fMaxY ) {
      fIh++;
      continue;
    }
    ret = fIh;
    fIh++;
    break;
  } while ( 1 );
  return ret;
}
