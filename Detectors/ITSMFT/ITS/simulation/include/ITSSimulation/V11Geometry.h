// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file V11Geometry.h
/// \brief Definition of the V11Geometry class

#ifndef ALICEO2_ITS_V11GEOMETRY_H_
#define ALICEO2_ITS_V11GEOMETRY_H_

#include <TMath.h>   // for DegToRad, Cos, Sin, Tan
#include <TObject.h> // for TObject
#include "Rtypes.h"  // for Double_t, Int_t, Bool_t, V11Geometry::Class, etc

class TGeoArb8;    // lines 11-11
class TGeoBBox;    // lines 16-16
class TGeoConeSeg; // lines 15-15
class TGeoPcon;    // lines 12-12
class TGeoTube;    // lines 13-13
class TGeoTubeSeg; // lines 14-14

namespace o2
{
namespace its
{

/// This class is a base class for the ITS geometry version 11. It contains common/standard
/// functions used in many places in defining the ITS geometry, version 11. Large posions of
/// the ITS geometry, version 11, should be derived from this class so as to make maximum
///  use of these common functions. This class also defines the proper conversion values such,
/// to cm and degrees, such that the most useful units, those used in the Engineering drawings,
/// can be used.
class V11Geometry : public TObject
{

 public:
  V11Geometry() : mDebug(){};

  V11Geometry(Int_t debug) : mDebug(debug){};

  ~V11Geometry()
    override = default;

  /// Sets the debug flag for debugging output
  void setDebug(Int_t level = 5)
  {
    mDebug = level;
  }

  /// Clears the debug flag so no debugging output will be generated
  void setNoDebug()
  {
    mDebug = 0;
  }

  /// Returns the debug flag value
  Bool_t getDebug(Int_t level = 1) const
  {
    return mDebug >= level;
  }

  // Static functions

  /// Define Trig functions for use with degrees (standerd TGeo angles).
  /// Sine function
  Double_t sinD(Double_t deg) const
  {
    return TMath::Sin(deg * TMath::DegToRad());
  }

  /// Cosine function
  Double_t cosD(Double_t deg) const
  {
    return TMath::Cos(deg * TMath::DegToRad());
  }

  /// Tangent function
  Double_t tanD(Double_t deg) const
  {
    return TMath::Tan(deg * TMath::DegToRad());
  }

  /// Determine the intersection of two lines
  /// Given the two lines, one passing by (x0,y0) with slope m and
  /// the other passing by (x1,y1) with slope n, returns the coordinates
  /// of the intersecting point (xi,yi)
  /// \param Double_t m The slope of the first line
  /// \param Double_t x0,y0 The x and y coord. of the first point
  /// \param Double_t n The slope of the second line
  /// \param Double_t x1,y1 The x and y coord. of the second point
  /// As an output it gives the coordinates xi and yi of the intersection point
  void intersectLines(Double_t m, Double_t x0, Double_t y0, Double_t n, Double_t x1, Double_t y1, Double_t& xi,
                      Double_t& yi) const;

  /// Determine the intersection of a line and a circle
  /// Given a line passing by (x0,y0) with slope m and a circle with
  /// radius rr and center (xc,yc), returns the coordinates of the
  /// intersecting points (xi1,yi1) and (xi2,yi2) (xi1 > xi2)
  /// \param Double_t m The slope of the line
  /// \param Double_t x0,y0 The x and y coord. of the point
  /// \param Double_t rr The radius of the circle
  /// \param Double_t xc,yc The x and y coord. of the center of circle
  /// As an output it gives the coordinates xi and yi of the intersection points
  /// Returns kFALSE if the line does not intercept the circle, otherwise kTRUE
  static Bool_t intersectCircle(Double_t m, Double_t x0, Double_t y0, Double_t rr, Double_t xc, Double_t yc,
                                Double_t& xi1, Double_t& yi1, Double_t& xi2, Double_t& yi2);

  /// Given the two points (x0,y0) and (x1,y1) and the location x, returns
  /// the value y corresponding to that point x on the line defined by the
  /// two points. Returns the value y corresponding to the point x on the line defined by
  /// the two points (x0,y0) and (x1,y1).
  /// \param Double_t x0 The first x value defining the line
  /// \param Double_t y0 The first y value defining the line
  /// \param Double_t x1 The second x value defining the line
  /// \param Double_t y1 The second y value defining the line
  /// \param Double_t x The x value for which the y value is wanted.
  Double_t yFrom2Points(Double_t x0, Double_t y0, Double_t x1, Double_t y1, Double_t x) const;

  /// Given the two points (x0,y0) and (x1,y1) and the location y, returns
  /// the value x corresponding to that point y on the line defined by the
  /// two points. Returns the value x corresponding to the point y on the line defined by
  /// the two points (x0,y0) and (x1,y1).
  /// \param Double_t x0 The first x value defining the line
  /// \param Double_t y0 The first y value defining the line
  /// \param Double_t x1 The second x value defining the line
  /// \param Double_t y1 The second y value defining the line
  /// \param Double_t y The y value for which the x value is wanted.
  Double_t xFrom2Points(Double_t x0, Double_t y0, Double_t x1, Double_t y1, Double_t y) const;

  /// Functions require at parts of Volume A to be already defined.
  /// Returns the value of Rmax corresponding to point z alone the line
  /// defined by the two points p.Rmax(i1),p-GetZ(i1) and p->GetRmax(i2),
  /// p->GetZ(i2).
  /// \param TGeoPcon *p The Polycone where the two points come from
  /// \param Int_t    i1 Point 1
  /// \param Int_t    i2 Point 2
  /// \param Double_t  z The value of z for which Rmax is to be found
  /// \param Double_t Rmx the value corresponding to z
  Double_t rMaxFrom2Points(const TGeoPcon* p, Int_t i1, Int_t i2, Double_t z) const;

  /// Returns the value of Rmin corresponding to point z alone the line
  /// defined by the two points p->GetRmin(i1),p->GetZ(i1) and
  /// p->GetRmin(i2),  p->GetZ(i2).
  /// \param TGeoPcon *p The Polycone where the two points come from
  /// \param Int_t    i1 Point 1
  /// \param Int_t    i2 Point 2
  /// \param Double_t  z The value of z for which Rmax is to be found
  /// \param Double_t Rmx the value corresponding to z
  Double_t rMinFrom2Points(const TGeoPcon* p, Int_t i1, Int_t i2, Double_t z) const;

  /// Returns the value of Rmin corresponding to point z alone the line
  /// defined by the two points p->GetRmin(i1),p->GetZ(i1) and
  /// p->GetRmin(i2), p->GetZ(i2). Returns the value r corresponding to z and the
  /// line defined by the two points
  /// \param Double_t az Array of z values
  /// \param Double_t  r Array of r values
  /// \param Int_t    i1 First Point in arrays
  /// \param Int_t    i2 Second Point in arrays
  /// \param Double_t z  Value z at which r is to be found
  Double_t rFrom2Points(const Double_t* ar, const Double_t* az, Int_t i1, Int_t i2, Double_t z) const;

  /// Returns the value of Z corresponding to point R alone the line
  /// defined by the two points p->GetRmin(i1),p->GetZ(i1) and
  /// p->GetRmin(i2),p->GetZ(i2). Returns the value z corresponding to r min
  /// and the line defined by the two points
  /// \param TGeoPcon *p The Poly cone where the two points come from.
  /// \param Int_t    i1 First Point in arrays
  /// \param Int_t    i2 Second Point in arrays
  /// \param Double_t r  Value r min at which z is to be found
  Double_t zFrom2MinPoints(const TGeoPcon* p, Int_t i1, Int_t i2, Double_t r) const;

  /// Returns the value of Z corresponding to point R alone the line
  /// defined by the two points p->GetRmax(i1),p->GetZ(i1) and
  /// p->GetRmax(i2),p->GetZ(i2). Returns the value z corresponding to
  /// r max and the line defined by the two points
  /// \param TGeoPcon *p The Poly cone where the two points come from.
  /// \param Int_t    i1 First Point in arrays
  /// \param Int_t    i2 Second Point in arrays
  /// \param Double_t r  Value r max at which z is to be found
  Double_t zFrom2MaxPoints(const TGeoPcon* p, Int_t i1, Int_t i2, Double_t r) const;

  /// Returns the value of z corresponding to point R alone the line
  /// defined by the two points p->GetRmax(i1),p->GetZ(i1) and
  /// p->GetRmax(i2),p->GetZ(i2). Returns the value z corresponding to r
  /// and the line defined by the two points
  /// \param Double_t z  Array of z values
  /// \param Double_t ar  Array of r values
  /// \param Int_t    i1  First Point in arrays
  /// \param Int_t    i2  Second Point in arrays
  /// \param Double_t r   Value r at which z is to be found
  Double_t zFrom2Points(const Double_t* az, const Double_t* ar, Int_t i1, Int_t i2, Double_t r) const;

  /// General Outer Cone surface equation Rmax
  /// Given 1 point from a TGeoPcon(z and Rmax) the angle tc returns r for
  /// a given z, an offset (distnace perpendicular to line at angle tc) of
  /// th may be applied. Returns the value Rmax corresponding to the line at angle th, offset by
  /// th, and the point p->GetZ/Rmin[ip] at the location z.
  /// \param TGeoPcon *p The poly cone where the initial point comes from
  /// \param Int_t    ip The index in p to get the point location
  /// \param Double_t tc The angle of that part of the cone is at
  /// \param Double_t  z The value of z to compute Rmax from
  /// \param Double_t th The perpendicular distance the parralell line is from the point ip
  Double_t rMaxFromZpCone(const TGeoPcon* p, int ip, Double_t tc, Double_t z, Double_t th = 0.0) const;

  // General Cone surface equation R(z). Returns the value R correstponding to the line at
  // angle th, offset by th, and the point p->GetZ/Rmax[ip] at the location z.
  // \param Double_t ar The array of R values
  // \param Double_t az The array of Z values
  // \param Int_t    ip The index in p to get the point location
  // \param Double_t tc The angle of that part of the cone is at
  // \param Double_t  z The value of z to compute R from
  // \param Double_t th The perpendicular distance the parralell line is from the point ip
  Double_t rFromZpCone(const Double_t* ar, const Double_t* az, int ip, Double_t tc, Double_t z,
                       Double_t th = 0.0) const;

  /// General Inner Cone surface equation Rmin.
  /// Given 1 point from a TGeoPcon(z and Rmin) the angle tc returns r for
  /// a given z, an offset (distnace perpendicular to line at angle tc) of
  /// th may be applied. Returns the value Rmin correstponding to the line at angle th,
  /// offset by th, and the point p->GetZ/Rmin[ip] at the location z.
  /// \param TGeoPcon  *p The poly cone where the initial point comes from
  /// \param Int_t     ip The index in p to get the point location
  /// \param Double_t  tc The angle of that part of the cone is at
  /// \param Double_t   z The value of z to compute Rmin from
  /// \param Double_t  th The perpendicular distance the parralell line is from the point ip
  Double_t rMinFromZpCone(const TGeoPcon* p, Int_t ip, Double_t tc, Double_t z, Double_t th = 0.0) const;

  /// General Outer cone Surface equation for z.
  /// Given 1 point from a TGeoPcon(z and Rmax) the angle tc returns z for
  /// a given Rmax, an offset (distnace perpendicular to line at angle tc) of
  /// th may be applied. Returns thevalue Z correstponding to the line at angle th,
  /// offset by th, and the point p->GetZ/Rmax[ip] at the location r.
  /// \param TGeoPcon *p The poly cone where the initial point comes from
  /// \param Int_t    ip The index in p to get the point location
  /// \param Double_t tc The angle of that part of the cone is at
  /// \param Double_t  r The value of Rmax to compute z from
  /// \param Double_t th The perpendicular distance the parralell line is from the point ip
  Double_t zFromRMaxpCone(const TGeoPcon* p, int ip, Double_t tc, Double_t r, Double_t th = 0.0) const;

  /// General Outer cone Surface equation for z.
  /// Returns the value Z correstponding to the line at angle th, offeset by
  /// th, and the point p->GetZ/Rmax[ip] at the locatin r.
  /// \param Double_t ar The array of R values
  /// \param Double_t az The array of Z values
  /// \param Int_t    ip The index in p to get the point location
  /// \param Double_t tc The angle of that part of the cone is at
  /// \param Double_t  r The value of Rmax to compute z from
  /// \param Double_t th The perpendicular distance the parralell line is from the point ip
  Double_t zFromRMaxpCone(const Double_t* ar, const Double_t* az, Int_t ip, Double_t tc, Double_t r,
                          Double_t th = 0.0) const;

  /// General Inner cone Surface equation for z.
  /// Given 1 point from a TGeoPcon(z and Rmin) the angle tc returns z for
  /// a given Rmin, an offset (distnace perpendicular to line at angle tc) of
  /// th may be applied. Returns the value Z correstponding to the line at angle th, offeset by
  /// th, and the point p->GetZ/Rmin[ip] at the location r.
  /// \param TGeoPcon *p The poly cone where the initial point comes from
  /// \param Int_t    ip The index in p to get the point location
  /// \param Double_t tc The angle of that part of the cone is at
  /// \param Double_t  r The value of Rmin to compute z from
  /// \param Double_t th The perpendicular distance the parralell line is from the point ip
  Double_t zFromRMinpCone(const TGeoPcon* p, int ip, Double_t tc, Double_t r, Double_t th = 0.0) const;

  /// Given two lines defined by the points i1, i2,i3 in the TGeoPcon
  /// class p that intersect at point p->GetZ(i2) return the point z,r
  /// that is Cthick away in the TGeoPcon class q. If points i1=i2
  /// and max == kTRUE, then p->GetRmin(i1) and p->GetRmax(i2) are used.
  /// if points i2=i3 and max=kTRUE then points p->GetRmax(i2) and
  /// p->GetRmin(i3) are used. If i2=i3 and max=kFALSE, then p->GetRmin(i2)
  /// and p->GetRmax(i3) are used.
  /// \param TGeoPcon *p Class where points i1, i2, and i3 are taken from
  /// \param Int_t    i1 First point in class p
  /// \param Int_t    i2 Second point in class p
  /// \param Int_t    i3 Third point in class p
  /// \param Double_t c Distance inside the outer/inner surface that the point j1
  /// is to be computed for
  /// \param TGeoPcon *q Pointer to class for results to be put into.
  /// \param Int_t    j1 Point in class q where data is to be stored.
  /// \param Bool_t   ma if kTRUE, then a Rmax value is computed, else a Rmin valule is computed
  /// \param TGeoPcon *q Pointer to class for results to be put into.
  void insidePoint(const TGeoPcon* p, Int_t i1, Int_t i2, Int_t i3, Double_t Cthick, TGeoPcon* q, Int_t j1,
                   Bool_t max) const;

  /// Given two intersecting lines defined by the points (x0,y0), (x1,y1) and
  /// (x1,y1), (x2,y2) {intersecting at (x1,y1)} the point (x,y) a distance
  /// c away is returned such that two lines a distance c away from the
  /// lines defined above intersect at (x,y).
  /// \param Double_t x0 X point on the first intersecting sets of lines
  /// \param Double_t y0 Y point on the first intersecting sets of lines
  /// \param Double_t x1 X point on the first/second intersecting sets of lines
  /// \param Double_t y1 Y point on the first/second intersecting sets of lines
  /// \param Double_t x2 X point on the second intersecting sets of lines
  /// \param Double_t y2 Y point on the second intersecting sets of lines
  /// \param Double_t c  Distance the two sets of lines are from each other
  /// \param Double_t x  X point for the intersecting sets of parellel lines
  /// \param Double_t y  Y point for the intersecting sets of parellel lines
  void insidePoint(Double_t x0, Double_t y0, Double_t x1, Double_t y1, Double_t x2, Double_t y2, Double_t c,
                   Double_t& x, Double_t& y) const;

  /// Given an initial point z0,r0, the initial angle theta0, and the radius
  /// of curvature, returns the point z1, r1 at the angle theta1. Theta
  /// measured from the r axis in the clock wise direction [degrees].
  /// \param Double_t rc     The radius of curvature
  /// \param Double_t theta0 The starting angle (degrees)
  /// \param Double_t z0     The value of z at theta0
  /// \param Double_t r0     The value of r at theta0
  /// \param Double_t theta1 The ending angle (degrees)
  /// \param Double_t &z1  The value of z at theta1
  /// \param Double_t &r1  The value of r at theta1
  void radiusOfCurvature(Double_t rc, Double_t theta0, Double_t z0, Double_t r0, Double_t theta1, Double_t& z1,
                         Double_t& r1) const;

  // Output functions for debugging

  /// Prints out the content of the TGeoArb8
  /// \param TGeoArb8 *a
  void printArb8(const TGeoArb8* a) const;

  /// Prints out the contents of the TGeoPcon
  /// \param TGeoPcon *a
  void printPcon(const TGeoPcon* a) const;

  /// Prints out the contents of the TGeoTube
  /// \param TGeoTube *a
  void printTube(const TGeoTube* a) const;

  /// Prints out the contents of the TGeoTubeSeg
  /// \param TGeoTubeSeg *a
  void printTubeSeg(const TGeoTubeSeg* a) const;

  /// Prints out the contents of the TGeoConeSeg
  /// \param TGeoConeSeg *a
  void printConeSeg(const TGeoConeSeg* a) const;

  /// Prints out the contents of the TGeoBBox
  /// \param TGeoBBox *a
  void printBBox(const TGeoBBox* a) const;

  /// Draws a cross sectional view of the TGeoPcon, Primarily for debugging.
  /// A TCanvas should exist first.
  /// \param TGeoPcon  *p The TGeoPcon to be "drawn"
  /// \param Int_t fillc The fill color to be used
  /// \param Int_t fills The fill style to be used
  /// \param Int_t linec The line color to be used
  /// \param Int_t lines The line style to be used
  /// \param Int_t linew The line width to be used
  /// \param Int_t markc The markder color to be used
  /// \param Int_t marks The markder style to be used
  /// \param Float_t marksize The marker size
  void drawCrossSection(const TGeoPcon* p, Int_t fillc = 7, Int_t fills = 4050, Int_t linec = 3, Int_t lines = 1,
                        Int_t linew = 4, Int_t markc = 2, Int_t marks = 4, Float_t marksize = 1.0) const;

  /// Computes the angles, t0 and t1 corresponding to the intersection of
  /// the line, defined by {x0,y0} {x1,y1}, and the circle, defined by
  /// its center {xc,yc} and radius r. If the line does not intersect the
  /// line, function returns kFALSE, otherwise it returns kTRUE. If the
  /// line is tangent to the circle, the angles t0 and t1 will be the same.
  /// Returns kTRUE if line intersects circle or kFALSE if line does not intersect circle
  /// or the line is not properly defined point {x0,y0} and {x1,y1} are the same point.
  /// \param Double_t x0 X of first point defining the line
  /// \param Double_t y0 Y of first point defining the line
  /// \param Double_t x1 X of Second point defining the line
  /// \param Double_t y1 Y of Second point defining the line
  /// \param Double_t xc X of Circle center point defining the line
  /// \param Double_t yc Y of Circle center point defining the line
  /// \param Double_t r radius of circle
  /// \param Double_t &t0 First angle where line intersects circle
  /// \param Double_t &t1 Second angle where line intersects circle
  Bool_t angleOfIntersectionWithLine(Double_t x0, Double_t y0, Double_t x1, Double_t y1, Double_t xc, Double_t yc,
                                     Double_t rc, Double_t& t0, Double_t& t1) const;

  /// Function to compute the ending angle, for arc 0, and starting angle,
  /// for arc 1, such that a straight line will connect them with no discontinuities.
  /// Begin_Html
  /*
      <img src="picts/ITS/V11Geometry_AnglesForRoundedCorners.gif">
     */
  // End_Html
  /// \param Double_t x0 X Coordinate of arc 0 center.
  /// \param Double_t y0 Y Coordinate of arc 0 center.
  /// \param Double_t r0 Radius of curvature of arc 0. For signe see figure.
  /// \param Double_t x1 X Coordinate of arc 1 center.
  /// \param Double_t y1 Y Coordinate of arc 1 center.
  /// \param Double_t r1 Radius of curvature of arc 1. For signe see figure.
  /// \param Double_t t0 Ending angle of arch 0, with respect to x axis, Degrees.
  /// \param Double_t t1 Starting angle of arch 1, with respect to x axis, Degrees.
  void anglesForRoundedCorners(Double_t x0, Double_t y0, Double_t r0, Double_t x1, Double_t y1, Double_t r1,
                               Double_t& t0, Double_t& t1) const;

  /// Define a general createMaterials function here so that if
  /// any specific subdetector does not define it this null function
  /// will due. This function is not declaired const so that a sub-
  /// detector's version may use class variables if they wish.
  /// Defined media here should correspond to the one defined in galice.cuts
  /// File which is red in (AliMC*) fMCApp::Init() { ReadTransPar(); }
  void createDefaultMaterials();

  virtual void createMaterials(){};

  /// Function to create the figure describing how the function
  /// anglesForRoundedCorners works.
  /// \param Double_t x0 X Coordinate of arc 0 center.
  /// \param Double_t y0 Y Coordinate of arc 0 center.
  /// \param Double_t r0 Radius of curvature of arc 0. For signe see figure.
  /// \param Double_t x1 X Coordinate of arc 1 center.
  /// \param Double_t y1 Y Coordinate of arc 1 center.
  /// \param Double_t r1 Radius of curvature of arc 1. For signe see figure.
  void makeFigure1(Double_t x0 = 0.0, Double_t y0 = 0.0, Double_t r0 = 2.0, Double_t x1 = -4.0, Double_t y1 = -2.0,
                   Double_t r1 = 1.0);

 protected:
  // Units, Convert from k?? to cm,degree,GeV,seconds,
  static const Double_t sMicron;  ///< Convert micron to TGeom's cm.
  static const Double_t sMm;      ///< Convert mm to TGeom's cm.
  static const Double_t sCm;      ///< Convert cm to TGeom's cm.
  static const Double_t sDegree;  ///< Convert degrees to TGeom's degrees
  static const Double_t sRadian;  ///< To Radians
  static const Double_t sGCm3;    ///< Density in g/cm^3
  static const Double_t sKgm3;    ///< Density in kg/m^3
  static const Double_t sKgdm3;   ///< Density in kg/dm^3
  static const Double_t sCelsius; ///< Temperature in degrees Celcius
  static const Double_t sPascal;  ///< Preasure in Pascal
  static const Double_t sKPascal; ///< Preasure in KPascal
  static const Double_t sEV;      ///< Energy in eV
  static const Double_t sKEV;     ///< Energy in KeV
  static const Double_t sMEV;     ///< Energy in MeV
  static const Double_t sGEV;     ///< Energy in GeV

 private:
  /// Basic function used to determine the ending angle and starting angles
  /// for rounded corners given the relative distance between the centers
  /// of the circles and the difference/sum of their radii. Case 0. Returns the angle in Degrees
  /// \param Double_t dx difference in x locations of the circle centers
  /// \param Double_t dy difference in y locations of the circle centers
  /// \param Double_t sdr difference or sum of the circle radii
  Double_t angleForRoundedCorners0(Double_t dx, Double_t dy, Double_t sdr) const;

  /// Basic function used to determine the ending angle and starting angles
  /// for rounded corners given the relative distance between the centers
  /// of the circles and the difference/sum of their radii. Case 0. Returns the angle in Degrees
  /// \param Double_t dx difference in x locations of the circle centers
  /// \param Double_t dy difference in y locations of the circle centers
  /// \param Double_t sdr difference or sum of the circle radii
  Double_t angleForRoundedCorners1(Double_t dx, Double_t dy, Double_t sdr) const;

  Int_t mDebug;                     //! Debug flag/level
  ClassDefOverride(V11Geometry, 1); // Base class for ITS v11 geometry
};
} // namespace its
} // namespace o2

#endif
