/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Modified : 22/06/2003 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
//
// FairGeoShapes
//
// Class to manage the GEANT/ROOT geometry shapes
//
// The 3D geometry in ROOT uses the ROOT library geom.
//
// This class holds a list of shape classes (type FairGeoBasicShape). They are
// used by all detectors when reading or writing geometry parameter containers
// to ascii files or to create an event display. This class is instantiated in
// the class HSpectrometer inside the function createDetGeomPar(Text_t* name).
// Every geometry parameter container gets a pointer to it.
//
// The individual shape classes are accessible with the selectShape(...)
// functions. The functions readPoints(...) and writePoints(...) use it
// internally.
//
// In the current version the following shapes are implemented:
//     BOX  TRAP  TRD1  PGON  PCON TUBE TUBS CONE CONS SPHE  ELTU
//     TORUS is added 06.11.06  m.al-turany
//
///////////////////////////////////////////////////////////////////////////////

#include "FairGeoShapes.h"

#include "FairGeoAssembly.h"
#include "FairGeoBasicShape.h"
#include "FairGeoBrik.h"
#include "FairGeoCone.h"
#include "FairGeoCons.h"
#include "FairGeoEltu.h"
#include "FairGeoPcon.h"
#include "FairGeoPgon.h"
#include "FairGeoSphe.h"
#include "FairGeoTorus.h"
#include "FairGeoTrap.h"
#include "FairGeoTrd1.h"
#include "FairGeoTube.h"
#include "FairGeoTubs.h"
#include "FairGeoVolume.h"

#include <TList.h>
#include <fstream>   // for fstream

ClassImp(FairGeoShapes);

FairGeoShapes::FairGeoShapes()
    : TObject()
    , shapes(new TList())
{
    // constructor creates empty list of shapes
}

FairGeoShapes::~FairGeoShapes()
{
    // destructor deletes all shapes
    shapes->Delete();
    delete shapes;
}

FairGeoBasicShape* FairGeoShapes::selectShape(FairGeoVolume* volu)
{
    // returns a pointer to the shape used in the given volume
    // calls internally selectShape(TString&) with the name of the shape
    // returns NULL if the corresponding shape class is not implemented
    const TString& name(volu->getShape());
    return selectShape(name);
}

FairGeoBasicShape* FairGeoShapes::selectShape(const TString& name)
{
    // returns a pointer to the shape given by name
    // creates a shape object and adds it to the list of shapes if
    // not existing
    // returns NULL if the corresponding shape class is not implemented
    TString allShapes[13] = {
        "BOX ", "TRAP", "TRD1", "PGON", "PCON", "TUBE", "TUBS", "CONE", "CONS", "SPHE", "ELTU", "TORUS", "ASSEMBLY"};
    TString sName(name);
    if (sName.Length() == 3) {
        sName += " ";
    }
    FairGeoBasicShape* s = static_cast<FairGeoBasicShape*>(shapes->FindObject(sName));
    if (s) {
        return s;
    }
    Int_t no = -1;
    for (Int_t i = 0; i < 13; i++) {
        if (sName.CompareTo(allShapes[i]) == 0) {
            no = i;
        }
    }
    switch (no) {
        case 0: {
            s = new FairGeoBrik();
            break;
        }
        case 1: {
            s = new FairGeoTrap();
            break;
        }
        case 2: {
            s = new FairGeoTrd1();
            break;
        }
        case 3: {
            s = new FairGeoPgon();
            break;
        }
        case 4: {
            s = new FairGeoPcon();
            break;
        }
        case 5: {
            s = new FairGeoTube();
            break;
        }
        case 6: {
            s = new FairGeoTubs();
            break;
        }
        case 7: {
            s = new FairGeoCone();
            break;
        }
        case 8: {
            s = new FairGeoCons();
            break;
        }
        case 9: {
            s = new FairGeoSphe();
            break;
        }
        case 10: {
            s = new FairGeoEltu();
            break;
        }
        case 11: {
            s = new FairGeoTorus();
            break;
        }
        case 12: {
            s = new FairGeoAssembly();
            break;
        }
        default: {
            Error("selectShape", "shape %s not implemented", name.Data());
        }
    }
    if (s) {
        shapes->Add(s);
    }
    return s;
}

Int_t FairGeoShapes::readPoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // reads the points of the given volume from the Ascii file
    // returns the number of points read
    // returns 0 if if the corresponding shape class is not implemented
    FairGeoBasicShape* s = selectShape(volu);
    if (s) {
        return s->readPoints(pFile, volu);
    } else {
        return 0;
    }
}

Bool_t FairGeoShapes::writePoints(std::fstream* pFile, FairGeoVolume* volu)
{
    // writes the points of the given volume to the Ascii file
    // return kFALSE if the corresponding shape class is not implemented
    FairGeoBasicShape* s = selectShape(volu);
    if (s) {
        return s->writePoints(pFile, volu);
    } else {
        return kFALSE;
    }
}

void FairGeoShapes::printPoints(FairGeoVolume* volu)
{
    // writes the points of the given volume to the Ascii file
    // return kFALSE if the corresponding shape class is not implemented
    FairGeoBasicShape* s = selectShape(volu);
    if (s) {
        return s->printPoints(volu);
    }
}
