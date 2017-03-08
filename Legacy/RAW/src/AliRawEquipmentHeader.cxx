/**************************************************************************
 * Copyright(c) 1998-2003, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

// @(#) $Id$
// Author: Fons Rademakers  26/11/99

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// AliRawEquipmentHeader                                                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <Bytes.h>

#include "AliRawEquipmentHeader.h"
#include "AliDAQ.h"

#include <Riostream.h>

using std::cout;
using std::endl;
ClassImp(AliRawEquipmentHeader)

//______________________________________________________________________________
AliRawEquipmentHeader::AliRawEquipmentHeader():
  fSize(0),
  fEquipmentType(0),
  fEquipmentID(0xffffffff),
  fBasicElementSizeType(0)
{
  // Default constructor
  for(Int_t i = 0; i < kAttributeWords; i++)
    fTypeAttribute[i] = 0;
}

//______________________________________________________________________________
UInt_t AliRawEquipmentHeader::SwapWord(UInt_t x) const
{
   // Swap the endianess of the integer value 'x'

   return (((x & 0x000000ffU) << 24) | ((x & 0x0000ff00U) <<  8) |
           ((x & 0x00ff0000U) >>  8) | ((x & 0xff000000U) >> 24));
}

//______________________________________________________________________________
void AliRawEquipmentHeader::Swap()
{
   // Swap equipment header data. There is no way to see if the data
   // has already been swapped. This method is only called when the
   // header is read from the DATE event builder (GDC).

   fSize                 = SwapWord(fSize);
   fEquipmentType        = SwapWord(fEquipmentType);
   fEquipmentID          = SwapWord(fEquipmentID);
   fBasicElementSizeType = SwapWord(fBasicElementSizeType);
   for (int i = 0; i < kAttributeWords; i++)
      fTypeAttribute[i] = SwapWord(fTypeAttribute[i]);
}

//______________________________________________________________________________
void AliRawEquipmentHeader::Reset()
{
  // Reset the contents of the equipment
  // header data
  fSize = fEquipmentType = fBasicElementSizeType = 0;
  fEquipmentID = 0xffffffff;

  for(Int_t i = 0; i < kAttributeWords; i++)
    fTypeAttribute[i] = 0;
}

//_____________________________________________________________________________
void AliRawEquipmentHeader::Print( const Option_t* opt ) const
{
  // Dumps the equipment header
  // fields

  cout << opt << "  Equipment size: " << fSize << endl;
  cout << opt << "  Equipment type: " << fEquipmentType << endl;
  Int_t ddlIndex;
  cout << opt << "  Equipment ID: " << fEquipmentID << " ( " << AliDAQ::DetectorNameFromDdlID(fEquipmentID,ddlIndex) << " )" << endl;
  cout << opt << "  Type attribute: " << fTypeAttribute[0] << "-" << fTypeAttribute[1] << "-" << fTypeAttribute[2] << endl;
  cout << opt << "  Basic element size type: " << fBasicElementSizeType << endl;
}
