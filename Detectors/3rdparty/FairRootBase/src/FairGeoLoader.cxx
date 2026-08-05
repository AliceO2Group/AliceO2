/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
/***************************************************************************
                          FairGeoLoader.cxx  -  description
                             -------------------
    begin                : Thu Jun 17 2004
    copyright            : (C) 2004 by ilse koenig , M. Al-turany
    email                : m.al-turany@gsi.de
 ***************************************************************************/
#include "FairGeoLoader.h"

#include "FairGeoInterface.h"     // for FairGeoInterface
#include "FairGeoRootBuilder.h"   // for FairGeoRootBuilder

#include <TGeoManager.h>   // for TGeoManager
#include <cstdlib>         // for NULL, exit
#include <iostream>        // for cout, endl, operator<<, etc
#include <string.h>        // for strncmp

using std::cout;
using std::endl;

FairGeoLoader* FairGeoLoader::fgInstance = 0;

FairGeoLoader* FairGeoLoader::Instance() { return fgInstance; }

FairGeoLoader::FairGeoLoader()
    : TNamed()
    , fInterface(NULL)
    , fGeoBuilder(NULL)
{
    fgInstance = this;
}

FairGeoLoader::FairGeoLoader(const char* Name, const char* title)
    : TNamed(Name, title)
    , fInterface(NULL)
    , fGeoBuilder(NULL)
{
    if (fgInstance) {
        Fatal("FairGeoLoader", "Singleton instance already exists.");
        return;
    }
    fgInstance = this;
    fInterface = new FairGeoInterface;
    if (strncmp(Name, "TGeo", 4) == 0) {
        TGeoManager* geom = new TGeoManager("FAIRGeom", "FAIR geometry");
        fGeoBuilder = new FairGeoRootBuilder("TGeo builder", "geometry builder");
        (static_cast<FairGeoRootBuilder*>(fGeoBuilder))->setGeoManager(geom);
    } else if (strncmp(Name, "G3Native", 8) == 0) {
        cout << "-I- FairGeoLoader() : Native G3 Geometry is used: This option is not supported any more!" << endl;
        exit(0);
        //        gGeoManager = NULL;
        //        fGeoBuilder=new FairGeoG3Builder("G3 builder","geometry builder");
    }

    fInterface->setGeomBuilder(fGeoBuilder);
}

FairGeoLoader::~FairGeoLoader() { delete fInterface; }

ClassImp(FairGeoLoader);
