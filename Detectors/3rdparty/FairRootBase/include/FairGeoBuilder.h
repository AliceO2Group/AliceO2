/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOBUILDER_H
#define FAIRGEOBUILDER_H

#include <Rtypes.h>   // for Int_t, etc
#include <TNamed.h>   // for TNamed

class FairGeoNode;
class FairGeoMedium;

/**
 * abstract base class for geometry builders
 * @author ilse koenig
 */
class FairGeoBuilder : public TNamed
{
  protected:
    Int_t nRot;   // number of defined rotations
    Int_t nMed;   // number of defined media

    FairGeoBuilder();
    FairGeoBuilder(const char* name, const char* title);

  public:
    virtual ~FairGeoBuilder() {}
    virtual Bool_t createNode(FairGeoNode*, Int_t hadFormat = 0) = 0;
    virtual Int_t createMedium(FairGeoMedium*) = 0;
    virtual void finalize() {}
    inline const Int_t& GetNMedia() const { return nMed; }
    inline void SetNMedia(const Int_t& nmed) { nMed = nmed; }
    ClassDef(FairGeoBuilder, 0);
};

#endif /* !FAIRGEOBUILDER_H */
