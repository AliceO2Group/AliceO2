/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- ö : Ilse Koenig
//*-- Modified : 10/11/2003 by Ilse Koenig

////////////////////////////////////////////////////////////////////////////////
//
//  Class for tracking medium ( includes also material )
//
////////////////////////////////////////////////////////////////////////////////
#include "FairGeoMedium.h"

#include <TString.h>   // for TString
#include <cmath>       // for pow, log
// IWYU pragma: no_include <architecture/i386/math.h>
#include <climits>    // for INT_MAX
#include <fstream>    // for fstream
#include <iostream>   // for cout
#include <stdlib.h>   // for abs

using std::cout;
using std::log;
using std::pow;

ClassImp(FairGeoMedium);

FairGeoMedium::FairGeoMedium(const char* name)
    : TNamed(name, "")
    , medId(0)
    , autoflag(1)
    , nComponents(0)
    , weightFac(0)
    , ca(nullptr)
    , cz(nullptr)
    , cw(nullptr)
    , density(0)
    , radLen(0)
    , sensFlag(0)
    , fldFlag(0)
    , fld(0)
    , epsil(0)
    , madfld(-1)
    , maxstep(-1)
    , maxde(-1)
    , minstep(-1)
    , npckov(0)
    , ppckov(nullptr)
    , absco(nullptr)
    , effic(nullptr)
    , rindex(nullptr){
          // Constructor for a medium with name and index id
          //  SetName(name);
      };

FairGeoMedium::~FairGeoMedium()
{
    // Destructor
    if (nComponents > 0) {
        delete[] ca;
        ca = 0;
        delete[] cz;
        cz = 0;
        delete[] cw;
        cw = 0;
        nComponents = 0;
    }
    if (npckov > 0) {
        delete[] ppckov;
        ppckov = 0;
        delete[] absco;
        absco = 0;
        delete[] effic;
        effic = 0;
        delete[] rindex;
        rindex = 0;
        npckov = 0;
    }
}

void FairGeoMedium::setNComponents(Int_t n)
{
    // Sets the number of components in the material
    if (n == 0) {
        return;
    }
    Int_t k = abs(n);
    if (nComponents != 0 && k != nComponents) {
        delete[] ca;
        delete[] cz;
        delete[] cw;
        nComponents = 0;
    }
    if (nComponents == 0) {
        nComponents = k;
        ca = new Double_t[k];
        cz = new Double_t[k];
        cw = new Double_t[k];
    }
    weightFac = static_cast<Int_t>(n / nComponents);
}

Bool_t FairGeoMedium::setComponent(Int_t i, Double_t a, Double_t z, Double_t weight)
{
    // Defines the ith material component
    if (i < 0 || i >= nComponents) {
        Error("setNComponents", "Wrong index");
        return kFALSE;
    }
    ca[i] = a;
    cz[i] = z;
    cw[i] = weight;
    return kTRUE;
}

void FairGeoMedium::getComponent(Int_t i, Double_t* p)
{
    // Returns the ith material component
    if (i >= 0 && i < nComponents) {
        p[0] = ca[i];
        p[1] = cz[i];
        p[2] = cw[i];
        //   cout << " -I p: " << p[0] << p[1] << p[2] << endl;
    } else {
        p[0] = p[1] = p[2] = 0.;
    }
}

void FairGeoMedium::setNpckov(Int_t n)
{
    // Sets the number of optical parameters for the tracking of Cerenkov light
    if (n != npckov && npckov > 0) {
        delete[] ppckov;
        delete[] absco;
        delete[] effic;
        delete[] rindex;
    }
    npckov = n;
    if (n > 0) {
        ppckov = new Double_t[npckov];
        absco = new Double_t[npckov];
        effic = new Double_t[npckov];
        rindex = new Double_t[npckov];
    }
}

Bool_t FairGeoMedium::setCerenkovPar(Int_t i, Double_t p, Double_t a, Double_t e, Double_t r)
{
    // Defines the ith parameter set of the optical parameters
    if (i < 0 || i >= npckov) {
        Error("setNpckov", "Wrong index");
        return kFALSE;
    }
    ppckov[i] = p;
    absco[i] = a;
    effic[i] = e;
    rindex[i] = r;
    return kTRUE;
}

void FairGeoMedium::getCerenkovPar(Int_t i, Double_t* p)
{
    // returns the ith parameter set of the optical parameters
    if (i >= 0 && i < npckov) {
        p[0] = ppckov[i];
        p[1] = absco[i];
        p[2] = effic[i];
        p[3] = rindex[i];
    } else {
        p[0] = p[1] = p[2] = p[3] = 0.;
    }
}

void FairGeoMedium::setMediumPar(Int_t sensitivityFlag,
                                 Int_t fieldFlag,
                                 Double_t maxField,
                                 Double_t precision,
                                 Double_t maxDeviation,
                                 Double_t maxStep,
                                 Double_t maxDE,
                                 Double_t minStep)
{
    // Sets the medium parameters
    sensFlag = sensitivityFlag;
    fldFlag = fieldFlag;
    fld = maxField;
    epsil = precision;
    madfld = maxDeviation;
    maxstep = maxStep;
    maxde = maxDE;
    minstep = minStep;
}

void FairGeoMedium::getMediumPar(Double_t* params)
{
    // Returns the medium parameters
    params[0] = sensFlag;
    params[1] = fldFlag;
    params[2] = fld;
    params[3] = madfld;
    params[4] = maxstep;
    params[5] = maxde;
    params[6] = epsil;
    params[7] = minstep;
    params[8] = 0.;
    params[9] = 0.;
}

void FairGeoMedium::read(std::fstream& fin, Int_t aflag)
{
    // Reads the parameters from file
    autoflag = aflag;
    Int_t n;
    fin >> n;
    setNComponents(n);
    for (Int_t ik = 0; ik < nComponents; ik++) {
        fin >> ca[ik];
    }
    for (Int_t i = 0; i < nComponents; i++) {
        fin >> cz[i];
    }
    fin >> density;
    if (nComponents == 1) {
        cw[0] = 1.;
        calcRadiationLength();
    } else {
        for (Int_t i = 0; i < nComponents; i++) {
            fin >> cw[i];
        }
    }
    fin >> sensFlag >> fldFlag >> fld >> epsil;
    if (autoflag < 1) {
        fin >> madfld >> maxstep >> maxde >> minstep;
    } else {
        // to use this feature one has to set TGeant3::SetAUTO(0), thus if the media does not
        // defined these values one can force Geant3 to calculate them by given them a value
        // of -1
        madfld = -1;
        maxstep = -1;
        maxde = -1;
        minstep = -1;
    }
    fin >> n;
    if (n > 0 && n < (INT_MAX - 1)) {
        setNpckov(n);
        //  if (n>0) {
        for (Int_t i = 0; i < n; i++) {
            fin >> ppckov[i] >> absco[i] >> effic[i] >> rindex[i];
        }
    }
}

void FairGeoMedium::print()
{
    // Prints the medium definition
    const char* bl = "  ";
    cout << GetName() << '\n' << nComponents * weightFac << bl;
    for (Int_t ii = 0; ii < nComponents; ii++) {
        cout << ca[ii] << bl;
    }
    for (Int_t j = 0; j < nComponents; j++) {
        cout << cz[j] << bl;
    }
    cout << density << bl;
    if (nComponents < 2) {
        cout << radLen;
    } else
        for (Int_t iik = 0; iik < nComponents; iik++) {
            cout << cw[iik] << bl;
        }
    cout << '\n' << sensFlag << bl << fldFlag << bl << fld << bl << epsil << '\n';
    if (autoflag < 1) {
        cout << madfld << bl << maxstep << bl << maxde << bl << minstep << '\n';
    }
    cout << npckov << '\n';
    if (npckov > 0) {
        for (Int_t i = 0; i < npckov; i++) {
            cout << ppckov[i] << bl << absco[i] << bl << effic[i] << bl << rindex[i] << '\n';
        }
    }
    cout << '\n';
}

void FairGeoMedium::write(std::fstream& fout)
{
    // Writes the medium definition into stream
    const char* bl = "  ";
    fout << GetName() << '\n' << nComponents * weightFac << bl;
    for (Int_t i = 0; i < nComponents; i++) {
        fout << ca[i] << bl;
    }
    for (Int_t j = 0; j < nComponents; j++) {
        fout << cz[j] << bl;
    }
    fout << density << bl;
    if (nComponents < 2) {
        fout << radLen;
    } else
        for (Int_t i = 0; i < nComponents; i++) {
            fout << cw[i] << bl;
        }
    fout << '\n' << sensFlag << bl << fldFlag << bl << fld << bl << epsil << '\n';
    if (autoflag < 1) {
        fout << madfld << bl << maxstep << bl << maxde << bl << minstep << '\n';
    }
    fout << npckov << '\n';
    if (npckov > 0) {
        for (Int_t i = 0; i < npckov; i++) {
            fout << ppckov[i] << bl << absco[i] << bl << effic[i] << bl << rindex[i] << '\n';
        }
    }
    fout << '\n';
}

Bool_t FairGeoMedium::calcRadiationLength()
{
    // calculates radiation length
    // formula in GEANT manual CONS110
    if (cz[0] < 1.e-6) {   // VAKUUM$
        radLen = 1.e+16;
        return kTRUE;
    }
    Double_t alpha = 1 / 137.;   // fine structure constant
    Double_t fac = .1912821;     // 4*((electron radius)**2)*(avogadro's number)
    Double_t z, a, w, az2, fc, y, xi, x0i, amol = 0., x0itot = 0.;
    if (weightFac > 0) {
        amol = 1.;
    } else {
        for (int i = 0; i < nComponents; i++) {
            amol += cw[i] * ca[i];
            if (amol == 0.) {
                Error("calcRadiationLength()", "amol==0 for medium %s", fName.Data());
                return kFALSE;
            }
        }
    }
    for (int i = 0; i < nComponents; i++) {
        z = cz[i];
        a = ca[i];
        if (weightFac > 0) {
            w = cw[i] / amol;
        } else {
            w = a * cw[i] / amol;
        }
        az2 = alpha * alpha * z * z;
        fc = az2 * (1. / (1. + az2) + 0.20206 - 0.0369 * az2 + 0.0083 * az2 * az2 - .002F * az2 * az2 * az2);
        y = log(183. / pow(z, 1. / 3.)) - fc;
        xi = static_cast<float>(log(1440. / pow(z, 2. / 3.)) / y);
        x0i = fac * alpha / a * z * (z + xi) * y;
        x0itot += (x0i * w);
    }
    if (x0itot == 0. || density == 0.) {
        Error("calcRadiationLength()", "x0itot=0 or density=0 for medium %s", fName.Data());
        return kFALSE;
    }
    radLen = 1 / density / x0itot;
    return kTRUE;
}
