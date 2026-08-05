/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Created : 10/11/03 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
// FairGeoInterface
//
// Manager class for geometry used by simulation

// *--15.12.04 M. Al-Turany
//  1-addInputFile(const char* name) is not used anymore
//  2-addGeoModule(FairGeoSet*pSet) is added
//  3-setMediaFile(const char *file) is added
//  3-findSet(const char* name) has been changed, no hard coding of name
//  4-nSets is now set by a method SetNoOfSets(Int_t) (see header file)

///////////////////////////////////////////////////////////////////////////////
#include "FairGeoInterface.h"

#include "FairGeoAsciiIo.h"   // for FairGeoAsciiIo
#include "FairGeoBuilder.h"   // for FairGeoBuilder
#include "FairGeoIo.h"        // for FairGeoIo
#include "FairGeoMedia.h"     // for FairGeoMedia
#include "FairGeoSet.h"       // for FairGeoSet
#include "FairGeoShapes.h"    // for FairGeoShapes

#include <TClass.h>       // for TClass
#include <TList.h>        // for TList
#include <TObjArray.h>    // for TObjArray
#include <iostream>       // for operator<<, basic_ostream, etc
#include <stdio.h>        // for sprintf
#include <string.h>       // for strcmp
#include <sys/select.h>   // for time_t
#include <time.h>         // for tm, localtime, time

using std::cout;
using std::endl;

ClassImp(FairGeoInterface);

FairGeoInterface::FairGeoInterface()
    : TObject()
    , fileInput(nullptr)
    , oraInput(nullptr)
    , output(nullptr)
    , nSets(0)
    , nActualSets(0)
    , sets(new TObjArray())
    , media(new FairGeoMedia())
    , shapes(new FairGeoShapes())
    , masterNodes(new TList())
    , setupFile("")
    , geoBuilder(nullptr)
{
    // Constructor
}

FairGeoInterface::~FairGeoInterface()
{
    // Destructor
    delete fileInput;
    fileInput = 0;
    delete oraInput;
    oraInput = 0;
    delete output;
    output = 0;

    //  if (geoBuilder) {
    //    delete geoBuilder;
    //    geoBuilder=0;
    //  }
    if (masterNodes) {
        masterNodes->Delete();
        delete masterNodes;
        masterNodes = 0;
    }
    if (sets) {
        sets->Delete();
        delete sets;
        sets = 0;
    }
    if (media) {
        delete media;
        media = 0;
    }
    if (shapes) {
        delete shapes;
        shapes = 0;
    }
}

void FairGeoInterface::addGeoModule(FairGeoSet* pSet)
{
    sets->AddLast(pSet);
    pSet->setShapes(shapes);
    pSet->setMasterNodes(masterNodes);
    nActualSets++;
}
void FairGeoInterface::setMediaFile(const char* file) { media->setInputFile(file); }

void FairGeoInterface::addInputFile(const char* file)
{

    Error("This Methode is not used any more ", "Use addGeoModule to add detectors: %s", file);
    /*
    // Adds the input file and creates the corresponding detector object
    FairGeoSet* pSet=0;
    TString setName(file);
    setName.ToLower();
    Int_t l=setName.Last('/')+1;
    setName=setName(l,setName.Length()-l);
    if (setName.Contains("media")) {
      media->setInputFile(file);
    } else {
      if (setName.Contains("cave")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoCave));
        if (!pSet) {
          pSet=new FairGeoCave;
          sets->AddAt(pSet,kFairGeoCave);
        }
      } else if (setName.Contains("pipe")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoPipe));
        if (!pSet) {
          pSet=new FairGeoPipe;
          sets->AddAt(pSet,kFairGeoPipe);
        }
      } else if (setName.Contains("target")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTarget));
        if (!pSet) {
          pSet=new FairGeoTarget;
          sets->AddAt(pSet,kFairGeoTarget);
        }
      } else if (setName.Contains("magnet")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoMagnet));
        if (!pSet) {
          pSet=new FairGeoMagnet;
          sets->AddAt(pSet,kFairGeoMagnet);
        }
      } else if (setName.Contains("rich")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoRich));
        if (!pSet) {
          pSet=new FairGeoRich;
          sets->AddAt(pSet,kFairGeoRich);
        }
      } else if (setName.Contains("sts")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoSts));
        if (!pSet) {
          pSet=new FairGeoSts;
          sets->AddAt(pSet,kFairGeoSts);
        }
      } else if (setName.Contains("trd")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTrd));
        if (!pSet) {
          pSet=new FairGeoTrd;
          sets->AddAt(pSet,kFairGeoTrd);
        }
      } else if (setName.Contains("tof")) {
        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTof));
        if (!pSet) {
          pSet=new FairGeoTof;
          sets->AddAt(pSet,kFairGeoTof);
        }
      } else Error("addInputFile","Unknown detector set: %s",file);
      if (pSet) {
        pSet->setShapes(shapes);
        pSet->setMasterNodes(masterNodes);
        pSet->setGeomFile(file);
        nActualSets++;
      }
    }
    */
}

FairGeoSet* FairGeoInterface::findSet(const char* name)
{
    // Find the detector set in the actual setup
    FairGeoSet* pSet = 0;
    TString setName(name);
    TString PSetName;
    setName.ToLower();
    Int_t l = setName.Last('/') + 1;
    setName = setName(l, setName.Length() - l);
    /*  if (setName.Contains("cave"))        pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoCave));
    else if (setName.Contains("pipe"))   pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoPipe));
    else if (setName.Contains("target")) pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTarget));
    else if (setName.Contains("magnet"))  pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoMagnet));
    else if (setName.Contains("rich"))   pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoRich));
    else if (setName.Contains("sts"))    pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoSts));
    else if (setName.Contains("trd"))    pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTrd));
    else if (setName.Contains("tof"))    pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoTof));
    else if (setName.Contains("ecal")) pSet=static_cast<FairGeoSet*>(sets->At(kFairGeoEcal));
    return pSet;
    */
    for (Int_t i = 0; i < nSets; i++) {
        PSetName = sets->At(i)->GetName();
        if (PSetName.Contains(setName.Data())) {
            pSet = static_cast<FairGeoSet*>(sets->At(i));
            cout << "FairGeoInterface::findSet" << pSet->GetName() << endl;
            break;
        }
    }
    return pSet;
}

Bool_t FairGeoInterface::readSet(FairGeoSet* pSet)
{
    // Reads the geometry for the detector part
    Bool_t rc = kFALSE;
    FairGeoIo* input = 0;
    if (pSet) {
        input = connectInput(pSet->getGeomFile());
        if (input) {
            rc = input->read(pSet, media);
        }
    }
    return rc;
}

Bool_t FairGeoInterface::writeSet(FairGeoSet* pSet)
{
    // Writes the geometry for the detector part to output
    if (pSet && connectOutput(pSet->GetName())) {
        return output->write(pSet);
    }
    return kFALSE;
}

Bool_t FairGeoInterface::writeSet(FairGeoSet* pSet, const char* filename)
{
    // Writes the geometry for the detector part to file
    if (pSet && output) {
        output->open(filename, "out");
        if (output->isOpen()) {
            return output->write(pSet);
        }
    }
    return kFALSE;
}

void FairGeoInterface::deleteSet(FairGeoSet* pSet)
{
    // Deletes the set
    if (pSet) {
        sets->Remove(pSet);
        delete pSet;
        pSet = 0;
    }
}

Bool_t FairGeoInterface::readMedia()
{
    // Reads the media
    Bool_t rc = kFALSE;
    FairGeoIo* input = connectInput(media->getInputFile());
    if (input) {
        rc = input->read(media);
    }
    return rc;
}

Bool_t FairGeoInterface::writeMedia()
{
    // Writes the media to output
    if (connectOutput("media")) {
        return output->write(media);
    }
    return kFALSE;
}

Bool_t FairGeoInterface::writeMedia(const char* filename)
{
    // Writes the media to file
    if (output) {
        output->open(filename, "out");
        if (output->isOpen()) {
            return output->write(media);
        }
    }
    return kFALSE;
}

Bool_t FairGeoInterface::readAll()
{
    // Reads the media and the whole geometry
    Bool_t rc = kTRUE;
    if (nActualSets == 0 && oraInput) {
        rc = oraInput->readGeomConfig(this);
        //  if (rc) addInputFile("media_gdb");
        if (rc) {
            addInputFile("media_gdb");
        }
    }
    if (rc) {
        rc = readSetupFile();
    }
    if (rc) {
        rc = readMedia();
    }
    FairGeoSet* pSet = 0;
    for (Int_t i = 0; i < nSets && rc; i++) {
        pSet = static_cast<FairGeoSet*>(sets->At(i));
        if (pSet) {
            cout << "Read " << pSet->GetName() << endl;
            rc = readSet(pSet);
        }
    }
    return rc;
}

Bool_t FairGeoInterface::writeAll()
{
    // Reads the media and the whole geometry to output
    Bool_t rc = kTRUE;
    FairGeoSet* pSet = 0;
    rc = writeMedia();
    for (Int_t i = 0; i < nSets && rc; i++) {
        pSet = static_cast<FairGeoSet*>(sets->At(i));
        if (pSet) {
            rc = writeSet(pSet);
        }
    }
    return rc;
}

Bool_t FairGeoInterface::createAll(Bool_t withCleanup)
{
    // Creates the whole geometry
    // (if withCleanup==kTRUE, the geometry input is deleted afterwards)
    Bool_t rc = kTRUE;
    Int_t n = 0;
    if (geoBuilder) {
        FairGeoSet* pSet = 0;
        for (Int_t i = 0; i < nSets && rc; i++) {
            pSet = static_cast<FairGeoSet*>(sets->At(i));
            if (pSet) {
                cout << "Create " << pSet->GetName() << endl;
                rc = pSet->create(geoBuilder);
                n++;
                if (withCleanup) {
                    sets->RemoveAt(i);
                    delete pSet;
                }
            }
        }
        if (rc && n > 0) {
            geoBuilder->finalize();
        }
    } else {
        Error("createAll()", "No Builder");
    }
    return rc;
}

Bool_t FairGeoInterface::createGeometry(Bool_t withCleanup)
{
    // Reads and creates the whole geometry
    // (if withCleanup==kTRUE, the geometry input is deleted after creation)
    Bool_t rc = kTRUE;
    Int_t n = 0;
    if (geoBuilder) {
        if (nActualSets == 0 && oraInput) {
            rc = oraInput->readGeomConfig(this);
            if (rc) {
                addInputFile("media_gdb");
            }
        }
        if (rc) {
            rc = readSetupFile();
        }
        if (rc) {
            rc = readMedia();
        }
        FairGeoSet* pSet = 0;
        for (Int_t i = 0; i < nSets && rc; i++) {
            pSet = static_cast<FairGeoSet*>(sets->At(i));
            if (pSet) {
                cout << "Read and create " << pSet->GetName() << endl;
                rc = readSet(pSet);
                if (rc) {
                    rc = pSet->create(geoBuilder);
                    n++;
                }
                if (withCleanup) {
                    sets->RemoveAt(i);
                    delete pSet;
                }
            }
        }
        if (rc && n > 0) {
            geoBuilder->finalize();
        }
    } else {
        Error("createAll()", "No Builder");
    }
    return rc;
}

FairGeoIo* FairGeoInterface::connectInput(const char* filename)
{
    // Connects the input (ASCII or Oracle)
    FairGeoIo* currentIo = 0;
    if (filename) {
        TString s(filename);
        s = s.Strip();
        if (s.EndsWith(".dat") || s.EndsWith(".geo") || s.EndsWith(".setup")) {
            if (!fileInput) {
                fileInput = new FairGeoAsciiIo();
            }
            fileInput->open(filename);
            currentIo = fileInput;
        } else if (s.EndsWith("_gdb")) {
            currentIo = oraInput;
        }
    }
    if (currentIo && currentIo->isOpen()) {
        return currentIo;
    }
    return 0;
}

Bool_t FairGeoInterface::connectOutput(const char* name)
{
    // Connects the output (ASCII or Oracle)
    if (output) {
        if (strcmp(output->IsA()->GetName(), "FairGeoAsciiIo") == 0) {
            TString fName(name);
            char buf[80];
            struct tm* newtime;
            time_t t;
            time(&t);
            newtime = localtime(&t);
            if (newtime->tm_mday < 10) {
                sprintf(buf, "_0%i", newtime->tm_mday);
            } else {
                sprintf(buf, "_%i", newtime->tm_mday);
            }
            fName = fName + buf;
            if (newtime->tm_mon < 9) {
                sprintf(buf, "0%i", newtime->tm_mon + 1);
            } else {
                sprintf(buf, "%i", newtime->tm_mon + 1);
            }
            fName = fName + buf;
            Int_t y = newtime->tm_year - 100;
            if (y < 10) {
                sprintf(buf, "0%i", y);
            } else {
                sprintf(buf, "%i", y);
            }
            fName = fName + buf;
            if (newtime->tm_hour < 10) {
                sprintf(buf, "0%i", newtime->tm_hour);
            } else {
                sprintf(buf, "%i", newtime->tm_hour);
            }
            fName = fName + buf;
            if (newtime->tm_min < 10) {
                sprintf(buf, "0%i", newtime->tm_min);
            } else {
                sprintf(buf, "%i", newtime->tm_min);
            }
            fName = fName + buf;
            if (newtime->tm_sec < 10) {
                sprintf(buf, "0%i", newtime->tm_sec);
            } else {
                sprintf(buf, "%i", newtime->tm_sec);
            }
            fName = fName + buf + ".geo";
            output->open(fName, "out");
            cout << "Output file for " << name << ":  " << (static_cast<FairGeoAsciiIo*>(output))->getFilename()
                 << endl;
        }
        if (output->isOpen() && output->isWritable()) {
            return kTRUE;
        }
    } else {
        Error("connectOutput", "No output open");
    }
    return kFALSE;
}

Bool_t FairGeoInterface::createSet(FairGeoSet* pSet)
{
    // Creates the geometry of detector part
    if (!geoBuilder) {
        return kFALSE;
    }
    return pSet->create(geoBuilder);
}

void FairGeoInterface::print()
{
    // Prints the media and the list of detector parts
    media->list();
    cout << "********************************************************************\n";
    cout << "List of detector parts:\n";
    FairGeoSet* pSet = 0;
    for (Int_t i = 0; i < nSets; i++) {
        pSet = static_cast<FairGeoSet*>(sets->At(i));
        if (pSet) {
            cout << "  " << pSet->GetName() << ":\t" << pSet->getGeomFile() << '\n';
        }
    }
    cout << "********************************************************************\n";
}

Bool_t FairGeoInterface::readGeomConfig(const char* configFile)
{
    // Reads the GEANT configuration file
    Bool_t rc = kFALSE;
    FairGeoIo* input = 0;
    if (configFile) {
        input = connectInput(configFile);
        if (input) {
            rc = input->readGeomConfig(this);
        }
    }
    return rc;
}

Bool_t FairGeoInterface::readSetupFile()
{
    // Reads the detector setups, needed for create only subsets
    Bool_t rc = kTRUE;
    if (setupFile.Length() > 0) {
        FairGeoAsciiIo* input = static_cast<FairGeoAsciiIo*>(connectInput(setupFile));
        if (input) {
            rc = input->readDetectorSetup(this);
        }
    }
    return rc;
}
