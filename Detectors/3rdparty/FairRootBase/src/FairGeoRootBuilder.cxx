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
// FairGeoRootBuilder
//
// Class to create the geometry in ROOT
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoRootBuilder.h"

#include "FairGeoMedium.h"      // for FairGeoMedium
#include "FairGeoNode.h"        // for FairGeoNode
#include "FairGeoRotation.h"    // for FairGeoRotation
#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector

#include <TArrayD.h>        // for TArrayD
#include <TGeoManager.h>    // for TGeoManager
#include <TGeoMaterial.h>   // for TGeoMixture, TGeoMaterial
#include <TGeoMatrix.h>     // for TGeoCombiTrans, etc
#include <TGeoMedium.h>     // for TGeoMedium
#include <TGeoVolume.h>     // for TGeoVolume, etc
#include <TString.h>        // for TString
#include <cstdio>           // for sprintf
#include <fstream>

ClassImp(FairGeoRootBuilder);

FairGeoRootBuilder::FairGeoRootBuilder()
    : FairGeoBuilder()
    , geoManager(nullptr)
{
    // Default constructor
}

FairGeoRootBuilder::FairGeoRootBuilder(const char* name, const char* title)
    : FairGeoBuilder(name, title)
    , geoManager(nullptr)
{
    // Constructor
}

Bool_t FairGeoRootBuilder::createNode(FairGeoNode* volu, Int_t hadFormat)
{
    // Creates the volume
    if (!geoManager || !volu) {
        return kFALSE;
    }
    //  volu->print();

    TString nodeName = volu->GetName();
    if (hadFormat == 1) {
        volu->setHadFormat(hadFormat);
        nodeName = nodeName(0, 4);
    } else {
        nodeName = (volu->GetName());
        Ssiz_t l = nodeName.Last('#');
        if (l > 0) {
            nodeName = nodeName(0, l);
        }
    }

    FairGeoNode* mother = volu->getMotherNode();
    if (!mother && !volu->isTopNode()) {
        Error("createNode", "Mother volume of %s not found\n", volu->GetName());
        return kFALSE;
    }

    TGeoVolume* rv = 0;
    FairGeoNode* cv = volu->getCopyNode();
    //  if (cv) cout<<"Copy of "<<cv->GetName()<<endl;
    if (cv) {
        rv = cv->getRootVolume();
    }
    if (!rv) {
        FairGeoMedium* lmedium = volu->getMedium();
        Int_t lnMed = lmedium->getMediumIndex();
        if (lnMed <= 0) {
            lnMed = createMedium(lmedium);
        }
        if (lnMed <= 0) {
            return kFALSE;
        }
        TArrayD* par = volu->getParameters();
        // for (Int_t k=0;k<par->GetSize();k++) cout<<par->At(k)<<" ";
        // cout<<endl;
        // cout << "Create Volume of Shape " << volu->getShape() << endl;
        if (volu->getShape().Contains("TORUS")) {   // Torus is missing in the TGeoManager::Volume
            // this should solve the problem tell it is implimented in root
            TGeoMedium* medium = geoManager->GetMedium(lnMed);
            rv = geoManager->MakeTorus(
                nodeName.Data(), medium, par->At(0), par->At(1), par->At(2), par->At(3), par->At(4));
            // cout << "Create Torus" << nodeName.Data() << endl;
        } else if (volu->getShape().Contains("ASSEMBLY")) {
            rv = geoManager->MakeVolumeAssembly(nodeName.Data());

        } else {
            rv = geoManager->Volume(nodeName.Data(), volu->getShape().Data(), lnMed, par->GetArray(), par->GetSize());
        }
        volu->setCreated();
        if (volu->isModule() && cv) {
            cv->setCreated();
            cv->setRootVolume(rv);
            cv->getPosition();
        }
    }
    if (!rv) {
        return kFALSE;
    }
    volu->setRootVolume(rv);
    if (volu->isTopNode()) {
        geoManager->SetTopVolume(rv);
    } else {
        FairGeoTransform* trans = volu->getPosition();
        const FairGeoRotation& rot = trans->getRotMatrix();
        const FairGeoVector& pos = trans->getTransVector();
        TGeoMatrix* tr = 0;
        if ((const_cast<FairGeoRotation&>(rot)).isUnitMatrix()) {
            tr = new TGeoTranslation(pos.getX(), pos.getY(), pos.getZ());
        } else {
            nRot++;
            char b[13];
            snprintf(b, sizeof(b), "R%i", nRot);
            TGeoRotation* r = new TGeoRotation(b);
            Double_t a[9];
            for (Int_t i = 0; i < 9; i++) {
                a[i] = rot(i);
            }
            r->SetMatrix(a);
            tr = new TGeoCombiTrans(pos.getX(), pos.getY(), pos.getZ(), r);
        }
        TGeoVolume* mo = mother->getRootVolume();
        if (!mo) {
            return kFALSE;
        }
        mo->AddNode(rv, volu->getCopyNo(), tr);
    }
    return kTRUE;
}

Int_t FairGeoRootBuilder::createMedium(FairGeoMedium* med)
{
    // Creates the medium
    if (!geoManager && !med) {
        return 0;
    }
    Int_t nComp = med->getNComponents();
    Int_t weightFac = med->getWeightFac();
    TGeoMaterial* material = 0;
    Double_t p[3];
    if (nComp == 1) {
        med->getComponent(0, p);
        material = new TGeoMaterial(med->GetName(), p[0], p[1], med->getDensity(), med->getRadiationLength());
        // Interaction length not defined!!!!!!
    } else {
        material = new TGeoMixture(med->GetName(), nComp, med->getDensity());
        Double_t sumWeights = 0.;
        if (weightFac < 0) {
            for (Int_t i = 0; i < nComp; i++) {
                med->getComponent(i, p);
                sumWeights += p[0] * p[2];
            }
        }
        for (Int_t i = 0; i < nComp; i++) {
            med->getComponent(i, p);
            if (weightFac > 0) {
                (static_cast<TGeoMixture*>(material))->DefineElement(i, p[0], p[1], p[2]);
            } else {
                (static_cast<TGeoMixture*>(material))->DefineElement(i, p[0], p[1], p[0] * p[2] / sumWeights);
            }
        }
    }
    nMed++;
    med->setMediumIndex(nMed);
    Double_t mp[10];
    med->getMediumPar(mp);
    TGeoMedium* medium = new TGeoMedium(med->GetName(), nMed, material, mp);
    if (medium) {
        return nMed;
    } else {
        return 0;
    }
}

void FairGeoRootBuilder::finalize()
{
    // Closes the geometry input in ROOT and draws the cave
    if (geoManager) {
        geoManager->CloseGeometry();
        geoManager->SetVisLevel(5);
        geoManager->GetTopVolume()->Draw();
    }
}

void FairGeoRootBuilder::checkOverlaps(Double_t ovlp)
{
    // Checks the geometry for overlaps and extrusions wit a default precision of 1 micron
    if (geoManager) {
        geoManager->CheckOverlaps(ovlp, "SAME");
        geoManager->PrintOverlaps();
    }
}
