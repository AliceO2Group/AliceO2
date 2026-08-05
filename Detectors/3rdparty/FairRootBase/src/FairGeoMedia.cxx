/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Last modified : 10/11/03 by Ilse Koenig

///////////////////////////////////////////////////////////////////////////////
// FairGeoMedia
//
// Class for materials and media
//
///////////////////////////////////////////////////////////////////////////////

#include "FairGeoMedia.h"

#include "FairGeoMedium.h"
#include "FairLogger.h"

#include <TList.h>
#include <iostream>

using std::cout;
using std::endl;

ClassImp(FairGeoMedia);

FairGeoMedia::FairGeoMedia()
    : TNamed()
    , media(new TList())
    , inputFile("")
    , author("")
    , description("")
{
    // Constructor
}

FairGeoMedia::~FairGeoMedia()
{
    // Destructor
    if (media) {
        media->Delete();
        delete media;
        media = 0;
    }
}

FairGeoMedium* FairGeoMedia::getMedium(const char* mediumName)
{
    // Returns the medium with name mediumName
    return static_cast<FairGeoMedium*>(media->FindObject(mediumName));
}

void FairGeoMedia::addMedium(FairGeoMedium* m)
{
    // Adds a medium to the list of media
    media->Add(m);
}

void FairGeoMedia::list()
{
    // Lists all media
    cout << "********************************************************************\n";
    cout << "List of media:\n";
    TListIter iter(media);
    FairGeoMedium* medium;
    while ((medium = static_cast<FairGeoMedium*>(iter.Next()))) {
        cout << "  " << medium->GetName() << "\n";
    }
}

void FairGeoMedia::print()
{
    // Prints the media
    if (!author.IsNull()) {
        cout << "//Author:      " << author << '\n';
    }
    if (!description.IsNull()) {
        cout << "//Description: " << description << '\n';
    }
    cout << "//----------------------------------------------------------\n";
    TListIter iter(media);
    FairGeoMedium* medium;
    Int_t i = 0;
    while ((medium = static_cast<FairGeoMedium*>(iter.Next()))) {
        if (medium->getAutoFlag() != 0) {
            medium->print();
            i++;
        }
    }
    if (i != media->GetSize()) {
        iter.Reset();
        cout << "//----------------------------------------------\n";
        cout << "AUTONULL\n";
        cout << "//----------------------------------------------\n";
        while ((medium = static_cast<FairGeoMedium*>(iter.Next()))) {
            if (medium->getAutoFlag() == 0) {
                medium->print();
            }
        }
    }
}

void FairGeoMedia::read(std::fstream& fin)
{
    // Reads the media from file
    LOG(info) << "FairGeoMedia: Read media ";
    const Int_t maxBuf = 256;
    char buf[maxBuf];
    Int_t autoflag = 1;
    while (!fin.eof()) {
        fin >> buf;
        if (buf[0] == '\0' || buf[0] == '/') {
            fin.getline(buf, maxBuf);
        } else if (fin.eof()) {
            break;
        } else {
            TString eleName(buf);
            if (eleName.CompareTo("AUTONULL") != 0) {
                FairGeoMedium* medium = new FairGeoMedium(eleName);
                medium->read(fin, autoflag);
                media->Add(medium);
            } else {
                autoflag = 0;
            }
        }
    }
}

void FairGeoMedia::write(std::fstream& fout)
{
    // Writes the media to file
    if (!author.IsNull()) {
        fout << "//Author:      " << author << '\n';
    }
    if (!description.IsNull()) {
        fout << "//Description: " << description << '\n';
    }
    fout << "//----------------------------------------------------------\n";
    TListIter iter(media);
    FairGeoMedium* medium;
    Int_t i = 0;
    while ((medium = static_cast<FairGeoMedium*>(iter.Next()))) {
        if (medium->getAutoFlag() != 0) {
            medium->write(fout);
            i++;
        }
    }
    if (i != media->GetSize()) {
        iter.Reset();
        fout << "//----------------------------------------------\n";
        fout << "AUTONULL\n";
        fout << "//----------------------------------------------\n";
        while ((medium = static_cast<FairGeoMedium*>(iter.Next()))) {
            if (medium->getAutoFlag() == 0) {
                medium->write(fout);
            }
        }
    }
}
