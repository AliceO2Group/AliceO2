/**************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN.          *
 * All rights reserved.                                       *
 *                                                            *
 * Author: The ALICE Off-line Project.                        *
 * Contributors are mentioned in the code where appropriate.  *
 *                                                            *
 * Permission to use, copy, modify and distribute this        *
 * software and its documentation strictly for non-commercial *
 * purposes is hereby granted without fee, provided that the  *
 * above copyright notice appears in all copies and that both *
 * the copyright notice and this permission notice appear in  *
 * the supporting documentation. The authors make no claims   *
 * about the suitability of this software for any purpose. It *
 * is provided "as is" without express or implied warranty.   *
 **************************************************************/
/* $Id$ */
//__________________________________________________________
// 
// Map of per strip Float_t information
// the floats are indexed by the coordinates 
//     DETECTOR # (1-3)
//     RING ID    ('I' or 'O', any case)
//     SECTOR #   (0-39)
//     STRIP #    (0-511)
//
// 
// Created Mon Nov  8 12:51:51 2004 by Christian Holm Christensen
// 
#include "AliFMDFloatMap.h"	//ALIFMDFLOATMAP_H
//__________________________________________________________
ClassImp(AliFMDFloatMap)
#if 0
  ; // This is here to keep Emacs for indenting the next line
#endif

//__________________________________________________________
AliFMDFloatMap::AliFMDFloatMap(const AliFMDMap& other)
  : AliFMDMap(other),
    fTotal(fMaxDetectors * fMaxRings * fMaxSectors * fMaxStrips),
    fData(0)
{
  if (fTotal == 0) fTotal = 51200;
  fData = new Float_t[fTotal];
  // Copy constructor
  if (!other.IsFloat()) return;
  for (Int_t i = 0; i < fTotal; i++) fData[i] = other.AtAsFloat(i);
}

//__________________________________________________________
AliFMDFloatMap::AliFMDFloatMap(const AliFMDFloatMap& other)
  : AliFMDMap(other.fMaxDetectors,
              other.fMaxRings,
              other.fMaxSectors,
              other.fMaxStrips),
    fTotal(fMaxDetectors * fMaxRings * fMaxSectors * fMaxStrips),
    fData(0)
{
  if (fTotal == 0) fTotal = 51200;
  fData = new Float_t[fTotal];
  // Copy constructor
  for (Int_t i = 0; i < fTotal; i++)
    fData[i] = other.fData[i];
}

//__________________________________________________________
AliFMDFloatMap::AliFMDFloatMap()
  : AliFMDMap(),
    fTotal(0),
    fData(0)
{
  // Constructor.
  // Parameters:
  //	None
}

//__________________________________________________________
AliFMDFloatMap::AliFMDFloatMap(Int_t maxDet,
			       Int_t maxRing,
			       Int_t maxSec,
			       Int_t maxStr)
  : AliFMDMap(maxDet, maxRing, maxSec, maxStr),
    fTotal(fMaxDetectors * fMaxRings * fMaxSectors * fMaxStrips),
    fData(0)
{
  // Constructor.
  // Parameters:
  //	maxDet	Maximum number of detectors
  //	maxRing	Maximum number of rings per detector
  //	maxSec	Maximum number of sectors per ring
  //	maxStr	Maximum number of strips per sector
  if (fTotal == 0) fTotal = 51200;
  fData = new Float_t[fTotal];
  Reset(0);
}

//__________________________________________________________
AliFMDFloatMap&
AliFMDFloatMap::operator=(const AliFMDFloatMap& other)
{
  // Assignment operator 
  if(&other != this){
    if(fMaxDetectors!= other.fMaxDetectors||
       fMaxRings    != other.fMaxRings||
       fMaxSectors  != other.fMaxSectors||
       fMaxStrips   != other.fMaxStrips){
      // allocate new memory only if the array size is different....
      fMaxDetectors = other.fMaxDetectors;
      fMaxRings     = other.fMaxRings;
      fMaxSectors   = other.fMaxSectors;
      fMaxStrips    = other.fMaxStrips;
      fTotal        = fMaxDetectors * fMaxRings * fMaxSectors * fMaxStrips;
      if (fTotal == 0) fTotal = 51200;
      if (fData) delete [] fData;
      fData = new Float_t[fTotal];
    }
    for (Int_t i = 0; i < fTotal; i++) fData[i] = other.fData[i];
  }
  return *this;
}


//__________________________________________________________
void
AliFMDFloatMap::Reset(const Float_t& val)
{
  // Reset map to val
  for (Int_t i = 0; i < fTotal; i++) fData[i] = val;
}

//__________________________________________________________
Float_t&
AliFMDFloatMap::operator()(UShort_t det, 
			   Char_t   ring, 
			   UShort_t sec, 
			   UShort_t str)
{
  // Get data
  // Parameters:
  //	det	Detector #
  //	ring	Ring ID
  //	sec	Sector #
  //	str	Strip #
  // Returns appropriate data
  return fData[CalcIndex(det, ring, sec, str)];
}

//__________________________________________________________
const Float_t&
AliFMDFloatMap::operator()(UShort_t det, 
			   Char_t   ring, 
			   UShort_t sec, 
			   UShort_t str) const
{
  // Get data
  // Parameters:
  //	det	Detector #
  //	ring	Ring ID
  //	sec	Sector #
  //	str	Strip #
  // Returns appropriate data
  return fData[CalcIndex(det, ring, sec, str)];
}

inline AliFMDFloatMap
operator*(const AliFMDMap& lhs, const AliFMDMap& rhs)
{
  AliFMDFloatMap r(lhs);
  r *= rhs;
  return r;
}
inline AliFMDFloatMap
operator/(const AliFMDMap& lhs, const AliFMDMap& rhs)
{
  AliFMDFloatMap r(lhs);
  r /= rhs;
  return r;
}
inline AliFMDFloatMap
operator+(const AliFMDMap& lhs, const AliFMDMap& rhs)
{
  AliFMDFloatMap r(lhs);
  r += rhs;
  return r;
}
inline AliFMDFloatMap
operator-(const AliFMDMap& lhs, const AliFMDMap& rhs)
{
  AliFMDFloatMap r(lhs);
  r -= rhs;
  return r;
}


//__________________________________________________________
// 
// EOF
// 

