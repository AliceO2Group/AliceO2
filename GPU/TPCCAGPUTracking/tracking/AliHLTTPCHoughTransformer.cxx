// @(#) $Id$
// origin hough/AliL3HoughBaseTransformer.cxx,v 1.16 Tue Mar 22 13:11:58 2005 UTC by cvetan 

// Author: Anders Vestbo <mailto:vestbo@fi.uib.no>
//*-- Copyright &copy ALICE HLT Group
//-------------------------------------------------------------------------
//          Implementation of the AliHLTTPCHoughTransformer class
//  that is the base class for AliHLTHoughTransformer,
//  AliHLTHoughTransformerVhdl, AliHLTHoughTransformerGlobal,
//  AliHLTHoughTransformerRow    
//-------------------------------------------------------------------------

#include "AliHLTTPCHoughTransformer.h"
#include "AliTPCRawStream.h"

/** \class AliHLTTPCHoughTransformer
<pre>
//_____________________________________________________________
// AliHLTTPCHoughTransformer
//
// The base class for implementations of Hough Transform on ALICE TPC data.
//
// This is an abstract class, and is only meant to provide the interface
// to the different implementations.
//
</pre>
*/

ClassImp(AliHLTTPCHoughTransformer)

AliHLTTPCHoughTransformer::AliHLTTPCHoughTransformer()
  :
  fLastTransformer(NULL),
  fSlice(0),
  fPatch(0),
  fLastPatch(-1),
  fNEtaSegments(0),
  fEtaMin(0),
  fEtaMax(0),
  fLowerThreshold(0),
  fUpperThreshold(1023),
  fDigitRowData(NULL),
  fZVertex(0.0)
{
  //Default constructor
}

AliHLTTPCHoughTransformer::AliHLTTPCHoughTransformer(Int_t slice,Int_t patch,Int_t netasegments,Float_t zvertex)
  :
  fLastTransformer(NULL),
  fSlice(0),
  fPatch(0),
  fLastPatch(-1),
  fNEtaSegments(0),
  fEtaMin(0),
  fEtaMax(0),
  fLowerThreshold(3),
  fUpperThreshold(1023),
  fDigitRowData(NULL),
  fZVertex(zvertex)
{
  //normal ctor
  Init(slice,patch,netasegments);
}

AliHLTTPCHoughTransformer::~AliHLTTPCHoughTransformer()
{
  //dtor
}

void AliHLTTPCHoughTransformer::Init(Int_t slice,Int_t patch,Int_t netasegments,Int_t /*n_seqs*/)
{
  //Transformer init
  fSlice = slice;
  fPatch = patch;
  fLastPatch = -1;
  fNEtaSegments = netasegments;
  fEtaMin = 0;
  fEtaMax = fSlice < 18 ? 1. : -1.;
}
