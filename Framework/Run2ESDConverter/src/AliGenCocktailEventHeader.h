#ifndef ALIGENCOCKTAILEVENTHEADER_H
#define ALIGENCOCKTAILEVENTHEADER_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/* $Id$ */

#include "AliGenEventHeader.h"


class AliGenCocktailEventHeader : public AliGenEventHeader
{
 public:
    AliGenCocktailEventHeader();
    AliGenCocktailEventHeader(const char* name);
    AliGenCocktailEventHeader(const AliGenCocktailEventHeader &header);
    virtual ~AliGenCocktailEventHeader();
    virtual void AddHeader(const AliGenEventHeader* header);
    virtual TList* GetHeaders() {return fHeaders;}
    virtual Int_t CalcNProduced();
    AliGenCocktailEventHeader & operator=(const AliGenCocktailEventHeader & rhs);
protected:
    TList  *fHeaders;     // List of Headers
    ClassDef(AliGenCocktailEventHeader,1)  // Event header for Cocktail event
};

#endif
