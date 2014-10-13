#ifndef ALIITSV11GEOMETRY_H
#define ALIITSV11GEOMETRY_H
/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               */

/*
  $Id$
 */

/*
  Base class for defining large parts of the ITS geometry, v11.
 */
#include <TObject.h>
#include <TMath.h>
#include "FairLogger.h"

class TGeoArb8;
class TGeoPcon;
class TGeoTube;
class TGeoTubeSeg;
class TGeoConeSeg;
class TGeoBBox;

class AliITSv11Geometry : public TObject {
  public:
    AliITSv11Geometry():fDebug() {};
    AliITSv11Geometry(Int_t debug):fDebug(debug) {};
    virtual ~AliITSv11Geometry(){};
    //
    // Sets the debug flag for debugging output
    void SetDebug(Int_t level=5){fDebug=level;}
    // Clears the debug flag so no debugging output will be generated
    void SetNoDebug(){fDebug=0;}
    // Returns the debug flag value
    Bool_t GetDebug(Int_t level=1)const {return fDebug>=level;}
    //
    // Static functions
    //
    // Define Trig functions for use with degrees (standerd TGeo angles).
    // Sine function
    Double_t SinD(Double_t deg)const{return TMath::Sin(deg*TMath::DegToRad());}
    // Cosine function
    Double_t CosD(Double_t deg)const{return TMath::Cos(deg*TMath::DegToRad());}
    // Tangent function
    Double_t TanD(Double_t deg)const{return TMath::Tan(deg*TMath::DegToRad());}
    // Determine the intersection of two lines
    void IntersectLines(Double_t m, Double_t x0, Double_t y0,
			Double_t n, Double_t x1, Double_t y1,
			Double_t &xi, Double_t &yi)const;
    // Determine the intersection of a line and a circle
    static Bool_t IntersectCircle(Double_t m, Double_t x0, Double_t y0,
			   Double_t rr, Double_t xc, Double_t yc,
				  Double_t &xi1, Double_t &yi1,
			   Double_t &xi2, Double_t &yi2);
    // Given the line, defined by the two points (x0,y0) and (x1,y1) and the
    // point x, return the value of y.
    Double_t Yfrom2Points(Double_t x0,Double_t y0,
                                 Double_t x1,Double_t y1,Double_t x)const;
    // Given the line, defined by the two points (x0,y0) and (x1,y1) and the
    // point y, return the value of x.
    Double_t Xfrom2Points(Double_t x0,Double_t y0,
                                 Double_t x1,Double_t y1,Double_t y)const;
    // Given 2 points from a TGeoPcon(z and Rmax) finds Rmax at given z
    Double_t RmaxFrom2Points(const TGeoPcon *p,Int_t i1,Int_t i2,
                                    Double_t z)const;
    // Given 2 points from a TGeoPcon(z and Rmin) finds Rmin at given z
    Double_t RminFrom2Points(const TGeoPcon *p,Int_t i1,Int_t i2,
                                    Double_t z)const;
    // Give two points in the array ar and az, returns the value r 
    // corresponding z along the line defined by those two points
    Double_t RFrom2Points(const Double_t *ar,const Double_t *az,
                                 Int_t i1,Int_t i2,Double_t z)const;
    // Given 2 points from a TGeoPcon(z and Rmax) finds z at given Rmin
    Double_t Zfrom2MinPoints(const TGeoPcon *p,Int_t i1,Int_t i2,
                                    Double_t r)const;
    // Given 2 points from a TGeoPcon(z and Rmax) finds z at given Rmax
    Double_t Zfrom2MaxPoints(const TGeoPcon *p,Int_t i1,Int_t i2,
                                    Double_t r)const;
    // Give two points in the array ar and az, returns the value z 
    // corresponding r along the line defined by those two points
    Double_t Zfrom2Points(const Double_t *az,const Double_t *ar,
                                 Int_t i1,Int_t i2,Double_t r)const;
    // Given 1 point from a TGeoPcon(z and Rmax) the angle tc returns r for 
    // a given z, an offset (distnace perpendicular to line at angle tc) of 
    // th may be applied.
    Double_t RmaxFromZpCone(const TGeoPcon *p,int ip,Double_t tc,
                                   Double_t z,Double_t th=0.0)const;
    Double_t RFromZpCone(const Double_t *ar,const Double_t *az,int ip,
                                Double_t tc,Double_t z,Double_t th=0.0)const;
    // Given 1 point from a TGeoPcon(z and Rmin) the angle tc returns r for 
    // a given z, an offset (distnace perpendicular to line at angle tc) of 
    // th may be applied.
    Double_t RminFromZpCone(const TGeoPcon *p,Int_t ip,Double_t tc,
                                   Double_t z,Double_t th=0.0)const;
    // Given 1 point from a TGeoPcon(z and Rmax) the angle tc returns z for 
    // a given Rmax, an offset (distnace perpendicular to line at angle tc) of 
    // th may be applied.
    Double_t ZFromRmaxpCone(const TGeoPcon *p,int ip,Double_t tc,
                                   Double_t r,Double_t th=0.0)const;
    // General Outer cone Surface equation for z.
    Double_t ZFromRmaxpCone(const Double_t *ar,const Double_t *az,
                                   Int_t ip,Double_t tc,Double_t r,
                                   Double_t th=0.0)const;
    // Given 1 point from a TGeoPcon(z and Rmin) the angle tc returns z for 
    // a given Rmin, an offset (distnace perpendicular to line at angle tc) of 
    // th may be applied.
    Double_t ZFromRminpCone(const TGeoPcon *p,int ip,Double_t tc,
                                   Double_t r,Double_t th=0.0)const;
    // Given two lines defined by the points i1, i2,i3 in the TGeoPcon 
    // class p that intersect at point p->GetZ(i2) return the point z,r 
    // that is Cthick away in the TGeoPcon class q. If points i1=i2
    // and max == kTRUE, then p->GetRmin(i1) and p->GetRmax(i2) are used.
    // if points i2=i3 and max=kTRUE then points p->GetRmax(i2) and
    // p->GetRmin(i3) are used. If i2=i3 and max=kFALSE, then p->GetRmin(i2)
    // and p->GetRmax(i3) are used.
    void InsidePoint(const TGeoPcon *p,Int_t i1,Int_t i2,Int_t i3,
                        Double_t Cthick,TGeoPcon *q,Int_t j1,Bool_t max)const;
    // Given two intersecting lines defined by the points (x0,y0), (x1,y1) and
    // (x1,y1), (x2,y2) {intersecting at (x1,y1)} the point (x,y) a distance
    // c away is returned such that two lines a distance c away from the
    // lines defined above intersect at (x,y).
     void InsidePoint(Double_t x0,Double_t y0,Double_t x1,Double_t y1,
                            Double_t x2,Double_t y2,Double_t c,
                            Double_t &x,Double_t &y)const;
    // Given a initial point z0,r0, the initial angle theta0, and the radius
    // of curvature, returns the point z1, r1 at the angle theta1. Theta
    // measured from the r axis in the clock wise direction [degrees].
    void RadiusOfCurvature(Double_t rc,Double_t theta0,Double_t z0,
                           Double_t r0,Double_t theta1,Double_t &z1,
                           Double_t &r1)const;
    //
    // Output functions for debugging
    //
    // Prints out the contents of the TGeoArb8
    void PrintArb8(const TGeoArb8 *a) const;
    // Prints out the contents of the TGeoPcon
    void PrintPcon(const TGeoPcon *a) const;
    // Prints out the contents of the TGeoTube
    void PrintTube(const TGeoTube *a) const;
    // Prints out the contents of the TGeoTubeSeg
    void PrintTubeSeg(const TGeoTubeSeg *a) const;
    // Prints out the contents of the TGeoConeSeg
    void PrintConeSeg(const TGeoConeSeg *a) const;
    // Prints out the contents of the TGeoBBox
    void PrintBBox(const TGeoBBox *a) const;
    // Draws a 2D crossection of the TGeoPcon r,z section
    void DrawCrossSection(const TGeoPcon *p,Int_t fillc=7,Int_t fills=4050,
                          Int_t linec=3,Int_t lines=1,Int_t linew=4,
                          Int_t markc=2,Int_t marks=4,
                          Float_t marksize=1.0) const;
    // Compute the angles where a line intersects a circle.
    Bool_t AngleOfIntersectionWithLine(Double_t x0,Double_t y0,
                                       Double_t x1,Double_t y1,
                                       Double_t xc,Double_t yc,
                                       Double_t rc,Double_t &t0,
                                       Double_t &t1)const;
    void AnglesForRoundedCorners(Double_t x0,Double_t y0,Double_t r0,
                                 Double_t x1,Double_t y1,Double_t r1,
                                 Double_t &t0,Double_t &t1)const;
    // Define a general CreateMaterials function here so that if
    // any specific subdetector does not define it this null function
    // will due. This function is not declaired const so that a sub-
    // detector's version may use class variables if they wish.
    void CreateDefaultMaterials();
    virtual void CreateMaterials(){};
    // Function to create figure needed for this class' documentation
    void MakeFigure1(Double_t x0=0.0,Double_t y0=0.0,Double_t r0=2.0,
                     Double_t x1=-4.0,Double_t y1=-2.0,Double_t r1=1.0);
  protected:

    // Units, Convert from k?? to cm,degree,GeV,seconds,
    static const Double_t fgkmicron; // Convert micron to TGeom's cm.
    static const Double_t fgkmm; // Convert mm to TGeom's cm.
    static const Double_t fgkcm; // Convert cm to TGeom's cm.
    static const Double_t fgkDegree; //Convert degrees to TGeom's degrees
    static const Double_t fgkRadian; //To Radians
    static const Double_t fgkgcm3;   // Density in g/cm^3
    static const Double_t fgkKgm3;   // Density in kg/m^3
    static const Double_t fgkKgdm3;   // Density in kg/dm^3
    static const Double_t fgkCelsius; // Temperature in degrees Celcius
    static const Double_t fgkPascal;  // Preasure in Pascal
    static const Double_t fgkKPascal;  // Preasure in KPascal
    static const Double_t fgkeV;  // Energy in eV
    static const Double_t fgkKeV;  // Energy in KeV
    static const Double_t fgkMeV;  // Energy in MeV
    static const Double_t fgkGeV;  // Energy in GeV

  private:
    Double_t AngleForRoundedCorners0(Double_t dx,Double_t dy,
                                     Double_t sdr)const;
    Double_t AngleForRoundedCorners1(Double_t dx,Double_t dy,
                                     Double_t sdr)const;
    Int_t fDebug; //! Debug flag/level
    ClassDef(AliITSv11Geometry,1) // Base class for ITS v11 geometry
};

#endif
