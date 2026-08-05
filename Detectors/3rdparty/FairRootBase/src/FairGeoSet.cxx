/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Created : 10/11/2003

/////////////////////////////////////////////////////////////
// FairGeoSet
//
// Base class for geometry of detector parts
//
/////////////////////////////////////////////////////////////
#include "FairGeoSet.h"

#include "FairGeoBasicShape.h"   // for FairGeoBasicShape
#include "FairGeoBuilder.h"      // for FairGeoBuilder
#include "FairGeoMedia.h"        // for FairGeoMedia
#include "FairGeoNode.h"         // for FairGeoNode, etc
#include "FairGeoShapes.h"       // for FairGeoShapes
#include "FairGeoTransform.h"    // for FairGeoTransform

#include <TArrayI.h>   // for TArrayI
#include <TString.h>   // for TString, operator<<
#include <ctype.h>     // for isalpha
#include <fstream>     // for fstream
#include <iostream>    // for cout
#include <string.h>    // for strcmp

class FairGeoMedium;

using std::cout;
using std::endl;
using std::ios;

ClassImp(FairGeoSet);

FairGeoSet::FairGeoSet()
    : TNamed()
    , hadesGeo(0)
    , volumes(new TList())
    , masterNodes(nullptr)
    , maxSectors(0)
    , maxKeepinVolumes(0)
    , maxModules(0)
    , modules(nullptr)
    , pShapes(nullptr)
    , geoFile("")
    , author("")
    , description("")
{
    // Constructor
}

FairGeoSet::~FairGeoSet()
{
    // Destructor
    delete volumes;
    volumes = 0;
    delete modules;
    modules = 0;
}

void FairGeoSet::setModules(Int_t s, Int_t* m)
{
    // Stores the modules given in 'm' as active modules in sector 's'
    // May be called with s=-1 for detectors not belonging to a sector
    if (!modules) {
        if (maxSectors == 0 && maxModules > 0) {
            modules = new TArrayI(maxModules);
        }
        if (maxSectors > 0 && maxModules > 0) {
            modules = new TArrayI(maxSectors * maxModules);
        }
    }
    Int_t sec = 0;
    if (s > 0) {
        sec = s;
    }
    for (Int_t i = 0; i < maxModules; i++) {
        if (modules) {
            modules->AddAt(m[i], (sec * maxModules + i));
        }
    }
}

Int_t* FairGeoSet::getModules()
{
    // Returns a linear array of all modules
    if (modules) {
        return modules->GetArray();
    }
    return 0;
}

Int_t FairGeoSet::getModule(Int_t s, Int_t m)
{
    // Returns 1, if the module is not explicitly set inactive
    // (in this case it returns 0)
    // May be called with s=-1 for detectors not belonging to a sector
    if (!modules) {
        return 1;
    }
    if (m < maxModules && s < 0) {
        return (*modules)[m];
    }
    if (m < maxModules && s < maxSectors) {
        return (*modules)[(s * maxModules + m)];
    }
    return 0;
}

Bool_t FairGeoSet::read(std::fstream& fin, FairGeoMedia* media)
{
    // Reads the geometry from file
    Int_t s1 = -1, s2 = 0;
    if (maxSectors > 0) {
        s1 = 0;
        s2 = maxSectors;
    }
    Bool_t rc = kTRUE;
    for (Int_t s = s1; s < s2 && rc; s++) {
        for (Int_t k = 0; k < maxKeepinVolumes && rc; k++) {
            TString keepinName = getKeepinName(s, k);
            rc = readKeepIn(fin, media, keepinName);
        }
    }
    for (Int_t m = 0; m < maxModules && rc; m++) {
        Bool_t containsActiveModule = kFALSE;
        for (Int_t s = s1; s < s2; s++) {
            if (getModule(s, m)) {
                containsActiveModule = kTRUE;
            }
        }
        TString modName = getModuleName(m);
        TString eleName = getEleName(m);
        rc = readModule(fin, media, modName, eleName, containsActiveModule);
    }
    return rc;
}

void FairGeoSet::readInout(std::fstream& fin)
{
    // Reads the inout flag (in old files)
    char c = ' ';
    do {
        fin.get(c);
    } while (c == ' ' || c == '\n');
    if (c != '0' && c != '1') {
        fin.putback(c);
    } else
        do {
            fin.get(c);
        } while (c != '\n');
    return;
}

void FairGeoSet::readTransform(std::fstream& fin, FairGeoTransform& tf)
{
    // Reads the transformation from file
    Double_t r[9], t[3];
    fin >> t[0] >> t[1] >> t[2];
    for (Int_t i = 0; i < 9; i++) {
        fin >> r[i];
    }
    tf.setRotMatrix(r);
    tf.setTransVector(t);
}

Bool_t FairGeoSet::readVolumeParams(std::fstream& fin, FairGeoMedia* media, FairGeoNode* volu, TList* refVolumes)
{
    // Reads the volume definition from file
    //  Int_t hadgeo =  1;
    if (volu == 0 || media == 0 || refVolumes == 0) {
        return kFALSE;
    }
    const Int_t maxbuf = 256;
    char buf[maxbuf];
    TString name(volu->GetName());
    Int_t nameLength = name.Length();
    Bool_t isCopy = kFALSE;
    fin >> buf;
    FairGeoNode* mother = 0;
    FairGeoCopyNode* refNode = 0;
    TString refName;

    if (volu->isKeepin()) {
        mother = getMasterNode(buf);
    } else if (volu->isModule()) {
        mother = getMasterNode(buf);
        if (!mother) {
            mother = getVolume(buf);
        }
        if (volu->isActive() && mother) {
            mother->setActive();
        }
    } else {
        mother = getVolume(buf);
        if (!mother) {
            mother = getMasterNode(buf);
        }
    }

    volu->setMother(mother);
    if (!mother) {
        Warning("readVolumeParams", "Mother volume %s not found!", buf);
    }

    if (hadesGeo == 1) {
        //  cout << " read copies in Hades format " << endl;

        if (nameLength > 4) {
            char c = ' ';
            do {
                fin.get(c);
            } while (c == ' ' || c == '\n');
            Int_t i = static_cast<Int_t>(c);
            fin.putback(c);

            if (!isalpha(i)) {
                isCopy = kTRUE;
            }
            refName = name(0, 4);
            refNode = static_cast<FairGeoCopyNode*>(refVolumes->FindObject(refName));
            if (!refNode) {
                if (isCopy) {
                    return kFALSE;
                }
                refVolumes->Add(new FairGeoCopyNode(refName.Data(), volu));
            } else {
                FairGeoNode* cn = refNode->pNode;
                volu->setCopyNode(cn);
                volu->setMedium(cn->getMedium());
                volu->setShape(cn->getShapePointer());
                Int_t n = cn->getNumPoints();
                volu->createPoints(n);
                for (Int_t k = 0; k < n; k++) {
                    volu->setPoint(k, *(cn->getPoint(k)));
                }
            }
        }

    } else {
        // cbm format
        Ssiz_t l = name.Last('#');
        if (l > 0) {
            char c;
            do {
                fin.get(c);
            } while (c == ' ' || c == '\n');
            Int_t i = static_cast<Int_t>(c);
            fin.putback(c);
            if (!isalpha(i)) {
                isCopy = kTRUE;
            }
            refName = name(0, l);
            refNode = static_cast<FairGeoCopyNode*>(refVolumes->FindObject(refName));
            if (!refNode) {
                if (isCopy) {
                    return kFALSE;
                }
                refVolumes->Add(new FairGeoCopyNode(refName.Data(), volu));
            } else {
                FairGeoNode* cn = refNode->pNode;
                volu->setCopyNode(cn);
                volu->setMedium(cn->getMedium());
                volu->setShape(cn->getShapePointer());
                Int_t n = cn->getNumPoints();
                volu->createPoints(n);
                for (Int_t k = 0; k < n; k++) {
                    volu->setPoint(k, *(cn->getPoint(k)));
                }
            }
        }
    }

    if (!isCopy) {
        fin >> buf;
        FairGeoBasicShape* sh = pShapes->selectShape(buf);
        if (sh) {
            volu->setShape(sh);
        } else {
            return kFALSE;
        }
        fin >> buf;
        FairGeoMedium* medium = media->getMedium(buf);
        if (!medium) {
            Error("readVolumeParams", "Medium %s not found in list of media", buf);
            return kFALSE;
        }
        volu->setMedium(medium);
        Int_t n = 0;
        fin.getline(buf, maxbuf);   // to read the end of line
        if (sh) {
            n = sh->readPoints(&fin, volu);
        }
        if (n <= 0) {
            return kFALSE;
        }
    }
    readTransform(fin, volu->getTransform());
    return kTRUE;
}

Bool_t FairGeoSet::readKeepIn(std::fstream& fin, FairGeoMedia* media, TString& name)
{
    // Reads the keepin volume from file
    fin.clear();
    fin.seekg(0, ios::beg);
    FairGeoNode* volu = 0;
    Bool_t rc = kTRUE;
    TList refVolumes;
    const Int_t maxbuf = 256;
    char buf[maxbuf];
    while (rc && !volu && !fin.eof()) {
        fin >> buf;
        if (buf[0] == '/') {
            fin.getline(buf, maxbuf);
        } else if (!fin.eof()) {
            if (strcmp(buf, name) == 0) {
                volu = new FairGeoNode;
                volu->SetName(buf);
                volu->setVolumeType(kFairGeoKeepin);
                readInout(fin);
                rc = readVolumeParams(fin, media, volu, &refVolumes);
            } else {
                do {
                    fin.getline(buf, maxbuf);
                } while (!fin.eof() && buf[0] != '/');
            }
        }
    }
    if (rc) {
        volumes->Add(volu);
    } else {
        delete volu;
        volu = 0;
    }
    refVolumes.Delete();
    return rc;
}

Bool_t FairGeoSet::readModule(std::fstream& fin,
                              FairGeoMedia* media,
                              TString& modName,
                              TString& eleName,
                              Bool_t containsActiveMod)
{
    // Reads the whole geometry of a module from file
    const Int_t maxbuf = 256;
    Text_t buf[maxbuf];
    fin.clear();
    fin.seekg(0, ios::beg);
    FairGeoNode* volu = 0;
    Bool_t rc = kTRUE;
    TList refVolumes;
    TString name;
    while (rc && !fin.eof()) {
        fin >> buf;
        if (buf[0] == '/') {
            fin.getline(buf, maxbuf);
        } else if (!fin.eof()) {
            // TString name(buf);
            name = buf;
            if (name.BeginsWith(modName)) {
                volu = new FairGeoNode;
                volu->SetName(buf);
                volu->setVolumeType(kFairGeoModule);
                Int_t a = getModule(getSecNumInMod(buf), getModNumInMod(buf));
                if (a > 0) {
                    volu->setActive(kTRUE);
                } else {
                    volu->setActive(kFALSE);
                }
                readInout(fin);
                rc = readVolumeParams(fin, media, volu, &refVolumes);
                if (rc) {
                    volumes->Add(volu);
                } else {
                    Error("readModule", "Failed for module %s\n", volu->GetName());
                    delete volu;
                    volu = 0;
                }
            } else if (!eleName.IsNull() && name.BeginsWith(eleName)) {
                volu = new FairGeoNode;
                volu->SetName(buf);
                volu->setVolumeType(kFairGeoElement);
                volu->setActive(containsActiveMod);
                rc = readVolumeParams(fin, media, volu, &refVolumes);
                if (rc) {
                    volumes->Add(volu);
                } else {
                    Error("readModule", "Failed for %s\n", volu->GetName());
                    delete volu;
                    volu = 0;
                }
            } else {
                do {
                    fin.getline(buf, maxbuf);
                } while (!fin.eof() && buf[0] != '/');
            }
        }
    }
    refVolumes.Delete();
    return rc;
}

void FairGeoSet::print()
{
    // Prints the geometry
    if (!author.IsNull()) {
        cout << "//Author:      " << author << '\n';
    }
    if (!description.IsNull()) {
        cout << "//Description: " << description << '\n';
    }
    if (!description.IsNull()) {
        cout << "//----------------------------------------------------------\n";
    }
    std::ios_base::fmtflags tmp = cout.setf(ios::fixed, ios::floatfield);
    TListIter iter(volumes);
    FairGeoNode* volu;
    while ((volu = static_cast<FairGeoNode*>(iter.Next()))) {
        volu->print();
    }
    cout.setf(tmp);
}

void FairGeoSet::write(std::fstream& fout)
{
    // Writes the volumes to file
    if (!author.IsNull()) {
        fout << "//Author:      " << author << '\n';
    }
    if (!description.IsNull()) {
        fout << "//Description: " << description << '\n';
    }
    fout << "//----------------------------------------------------------\n";
    std::ios_base::fmtflags tmp = fout.setf(ios::fixed, ios::floatfield);
    TListIter iter(volumes);
    FairGeoNode* volu;
    Bool_t rc = kTRUE;
    while ((volu = static_cast<FairGeoNode*>(iter.Next())) && rc) {
        rc = volu->write(fout);
    }
    fout.setf(tmp);
}

Bool_t FairGeoSet::create(FairGeoBuilder* builder)
{
    // Creates the geometry
    if (!builder) {
        return kFALSE;
    }
    TListIter iter(volumes);
    FairGeoNode* volu;
    Bool_t rc = kTRUE;
    while ((volu = static_cast<FairGeoNode*>(iter.Next())) && rc) {
        if (volu->isActive()) {

            if (hadesGeo == 1) {
                rc = builder->createNode(volu, hadesGeo);
            } else {
                rc = builder->createNode(volu);
            }
            if (rc) {
                if (volu->isTopNode() || volu->isRefNode()) {
                    FairGeoNode* n = getMasterNode(volu->GetName());
                    if (n) {
                        n->setCenterPosition(volu->getCenterPosition());
                        n->setRootVolume(volu->getRootVolume());
                    }
                }
            } else {
                Error("create", "Creation of %s failed!", volu->GetName());
            }
        }
    }
    return rc;
}

void FairGeoSet::compare(FairGeoSet& rset)
{
    // Compares the geometry parameters and print a diagnose
    if (fName.CompareTo(rset.GetName()) != 0) {
        Error("compare", "Sets have different type");
        return;
    }
    TListIter iter(volumes);
    FairGeoNode* volu;
    Int_t n = 0, nnf = 0, nDiff = 0;
    cout << "name\t mother medium shape  points    pos    rot\n";
    cout << "------------------------------------------------\n";
    while ((volu = static_cast<FairGeoNode*>(iter.Next()))) {
        n++;
        FairGeoNode* rNode = rset.getVolume(volu->GetName());
        if (!rNode) {
            nnf++;
        } else if (volu->compare(*rNode) > 0) {
            nDiff++;
        }
    }
    cout << "Number of volumes in first list:             " << n << '\n';
    cout << "Number of different volumes:                 " << nDiff << endl;
    cout << "Number of volumes not found in second list:  " << nnf << '\n';
    TList* rlist = rset.getListOfVolumes();
    if (rlist->GetSize() != (n - nnf)) {
        nnf = 0;
        TListIter riter(rlist);
        while ((volu = static_cast<FairGeoNode*>(riter.Next()))) {
            FairGeoNode* rNode = getVolume(volu->GetName());
            if (!rNode) {
                nnf++;
            }
        }
    } else {
        nnf = 0;
    }
    cout << "Number of additional volumes in second list: " << nnf << '\n';
}
