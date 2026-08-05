/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOROOTBUILDER_H
#define FAIRGEOROOTBUILDER_H

#include "FairGeoBuilder.h"   // for FairGeoBuilder

#include <Rtypes.h>   // for FairGeoRootBuilder::Class, etc

class FairGeoMedium;
class FairGeoNode;
class TGeoManager;

/**
 * class to create geometry in ROOT
 * @author M. Al-Turany
 */
class FairGeoRootBuilder : public FairGeoBuilder
{
  private:
    FairGeoRootBuilder(const FairGeoRootBuilder&);
    FairGeoRootBuilder& operator=(const FairGeoRootBuilder&);

  protected:
    TGeoManager* geoManager;   // ROOT geometry manager

  public:
    FairGeoRootBuilder();
    FairGeoRootBuilder(const char*, const char*);
    ~FairGeoRootBuilder() {}
    void setGeoManager(TGeoManager* me) { geoManager = me; }
    Bool_t createNode(FairGeoNode*, Int_t hadFormat = 0);
    Int_t createMedium(FairGeoMedium*);
    void finalize();
    void checkOverlaps(Double_t ovlp = 0.0001);
    ClassDef(FairGeoRootBuilder, 0);   //
};

#endif /* !FAIRGEOROOTBUILDER_H */
