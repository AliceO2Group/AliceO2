// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Cluster.cxx
/// \brief Implementation of the ITSMFT cluster

#include "ITSMFTReconstruction/Cluster.h"
#include "FairLogger.h"

#include <TMath.h>
#include <TString.h>

#include <cstdlib>

using namespace o2::ITSMFT;

ClassImp(o2::ITSMFT::Cluster)

#ifdef _ClusterTopology_
//______________________________________________________________________________
void Cluster::resetPattern()
{
  // reset pixels pattern
  memset(mPattern, 0, kMaxPatternBytes * sizeof(UChar_t));
}

//______________________________________________________________________________
Bool_t Cluster::testPixel(UShort_t row, UShort_t col) const
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return kFALSE;
  int bytn = nbits >> 3; // 1/8
  int bitn = 7 - (nbits % 8);
  return (mPattern[bytn] & (0x1 << bitn)) != 0;
  //
}

//______________________________________________________________________________
void Cluster::setPixel(UShort_t row, UShort_t col, Bool_t fired)
{
  // test if pixel at relative row,col is fired
  int nbits = row * getPatternColSpan() + col;
  if (nbits >= kMaxPatternBits)
    return;
  int bytn = nbits >> 3; // 1/8
  int bitn = 7 - (nbits % 8);
  if (nbits >= kMaxPatternBits)
    exit(1);
  if (fired)
    mPattern[bytn] |= (0x1 << bitn);
  else
    mPattern[bytn] &= (0xff ^ (0x1 << bitn));
  //
}

//______________________________________________________________________________
void Cluster::setPatternRowSpan(UShort_t nr, Bool_t truncated)
{
  // set pattern span in rows, flag if truncated
  mPatternNRows = kSpanMask & nr;
  if (truncated)
    mPatternNRows |= kTruncateMask;
}

//______________________________________________________________________________
void Cluster::setPatternColSpan(UShort_t nc, Bool_t truncated)
{
  // set pattern span in columns, flag if truncated
  mPatternNCols = kSpanMask & nc;
  if (truncated)
    mPatternNCols |= kTruncateMask;
}

#endif
/*
//______________________________________________________________________________
Bool_t Cluster::hasCommonTrack(const Cluster* cl) const
{
  // check if clusters have common tracks
  for (int i = 0; i < maxLabels; i++) {
    Label lbi = getLabel(i);
    if ( lbi.isEmpty() ) break;
    if ( !lbi.isPosTrackID() ) continue;

    for (int j = 0; j < maxLabels; j++) {
      Label lbj = cl->getLabel(j);
      if ( lbj.isEmpty() ) break;
      if ( !lbj.isPosTrackID() ) continue;
      if (lbi == lbj) return kTRUE;
    }
  }
  return kFALSE;
}
*/
//______________________________________________________________________________
void Cluster::print() const
{
  // print itself
  printf("Sensor %5d, nRow:%3d nCol:%3d n:%d |Err^2:%.3e %.3e %+.3e |",getSensorID(),getNx(),getNz(),
         getNPix(),getSigmaY2(),getSigmaZ2(),getSigmaYZ());
  printf("XYZ: %+.4e %+.4e %+.4e\n",getX(),getY(),getZ());
  //
 #ifdef _ClusterTopology_
  int nr = getPatternRowSpan();
  int nc = getPatternColSpan();
  printf("Pattern: %d rows from %d",nr,mPatternRowMin);
  if (isPatternRowsTruncated()) printf("(truncated)");
  printf(", %d cols from %d",nc,mPatternColMin);
  if (isPatternColsTruncated()) printf("(truncated)");
  printf("\n");
  for (int ir=0;ir<nr;ir++) {
    for (int ic=0;ic<nc;ic++) printf("%c",testPixel(ir,ic) ? '+':'-');
    printf("\n");
  }
#endif

}
