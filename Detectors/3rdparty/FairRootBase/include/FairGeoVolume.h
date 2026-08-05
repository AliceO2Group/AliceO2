/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOVOLUME_H
#define FAIRGEOVOLUME_H

#include "FairGeoMedium.h"      // for FairGeoMedium
#include "FairGeoTransform.h"   // for FairGeoTransform
#include "FairGeoVector.h"      // for FairGeoVector

#include <Rtypes.h>      // for Int_t, Text_t, Double_t, etc
#include <TNamed.h>      // for TNamed
#include <TObjArray.h>   // for TObjArray
#include <TString.h>     // for TString
#include <stddef.h>      // for NULL

/**
 * basic geometry parameters of a volume
 * @author Ilse koenig
 */
class FairGeoVolume : public TNamed
{
  private:
    FairGeoVolume& operator=(const FairGeoVolume&);

  protected:
    TString shape;                  /**eant shape of the volume*/
    TString mother;                 /** Name of the mother volume*/
    TObjArray* points;              /** array of points (type FairGeoVector)*/
    FairGeoTransform transform;     /** transformation relative to the mother system */
    FairGeoTransform fLabTransform; /** transformation to the LAB system; */
    FairGeoMedium fMedium;          /** medium properties */
    Int_t nPoints;                  /** number of points */
    Int_t fHadFormat;               //
    Int_t fgMCid;                   /** Volume Id given by MC*/

  public:
    inline FairGeoVolume();
    FairGeoVolume(FairGeoVolume&);
    virtual ~FairGeoVolume();
    const TString& getName() const { return fName; }
    const TString& getShape() const { return shape; }
    const TString& getMother() const { return mother; }
    FairGeoTransform& getTransform() { return transform; }
    FairGeoTransform& getLabTransform() { return fLabTransform; }
    FairGeoMedium& getMedium() { return fMedium; }
    Int_t getNumPoints() { return nPoints; }
    inline FairGeoVector* getPoint(const Int_t n);
    inline void setName(const Text_t* s);
    void setVolumePar(FairGeoVolume&);
    inline void setShape(const Text_t* s);
    inline void setMother(const Text_t* s);
    void createPoints(const Int_t);
    void setPoint(const Int_t, const Double_t, const Double_t, const Double_t);
    void setPoint(const Int_t, const FairGeoVector&);
    virtual void clear();
    virtual void print();
    virtual void setHadFormat(Int_t i) { fHadFormat = i; }
    Double_t getVolParameter(Int_t nPoint, Int_t pos);
    Int_t getMCid() { return fgMCid; }
    void setMCid(Int_t MCid) { fgMCid = MCid; }

    ClassDef(FairGeoVolume, 1);   //
};

// -------------------- inlines --------------------------

inline FairGeoVolume::FairGeoVolume()
    : TNamed()
    , shape("")
    , mother("")
    , points(NULL)
    , transform(FairGeoTransform())
    , fLabTransform(FairGeoTransform())
    , fMedium(0)
    , nPoints(0)
    , fHadFormat(0)
    , fgMCid(0)
{}

inline FairGeoVector* FairGeoVolume::getPoint(const Int_t n)
{
    if (points && n < nPoints) {
        return static_cast<FairGeoVector*>(points->At(n));
    } else {
        return 0;
    }
}

inline void FairGeoVolume::setName(const Text_t* s)
{
    fName = s;
    fName.ToUpper();
}

inline void FairGeoVolume::setShape(const Text_t* s)
{
    shape = s;
    shape.ToUpper();
}

inline void FairGeoVolume::setMother(const Text_t* s)
{
    mother = s;
    mother.ToUpper();
}

#endif /* !FAIRGEOVOLUME_H */
