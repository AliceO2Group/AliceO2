/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
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

/* $Id$ */

#include "AliGenCocktailEventHeader.h"
#include "AliGenEventHeader.h"
#include <TList.h>

ClassImp(AliGenCocktailEventHeader)


AliGenCocktailEventHeader::AliGenCocktailEventHeader():
    fHeaders(0)
{
// Default Constructor
}

AliGenCocktailEventHeader::AliGenCocktailEventHeader(const char* name):AliGenEventHeader(name),
    fHeaders(0)
{
// Constructor
}

AliGenCocktailEventHeader::AliGenCocktailEventHeader(const AliGenCocktailEventHeader &header):
    AliGenEventHeader(header),
    fHeaders(0)
{
// Copy Constructor
    header.Copy(*this);
}

AliGenCocktailEventHeader::~AliGenCocktailEventHeader()
{
// Constructor
    if (fHeaders) {
	fHeaders->Delete();
	delete fHeaders;
    }
}

void AliGenCocktailEventHeader::AddHeader(const AliGenEventHeader* header)
{
// Add a header to the list
    if (!fHeaders) fHeaders = new TList();
    fHeaders->Add(header->Clone());
}

Int_t AliGenCocktailEventHeader::CalcNProduced()
{
    // Calculate the total number of produced and stored particles
    TIter next(fHeaders);
    AliGenEventHeader *entry;
    //
    Int_t np = 0;
    while((entry = (AliGenEventHeader*)next())) {np += (entry->NProduced());}
    fNProduced = np;
    return (np);
}

AliGenCocktailEventHeader& AliGenCocktailEventHeader::operator=(const  AliGenCocktailEventHeader& rhs)
{
// Assignment operator
    rhs.Copy(*this); 
    return (*this);
}

