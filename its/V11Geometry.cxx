/**************************************************************************
 * Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 *                                                                        *
 * Author: The ALICE Off-line Project.                                    *
 * Contributors are mentioned in the code where appropriate.              *
 *                                                                        *
 * Permission to use, copy, modify and distribute this software and its   *
 * documentation strictly for non-commercial purposes is hereby granted   *
 * without fee, provided that the above copyright notice appears in all   *
 * copies and that both the copyright notice and this permission notice   *
 * appear in the supporting documentation. The authors make no claims     *
 * about the suitability of this software for any purpose. It is          *
 * provided "as is" without express or implied warranty.                  *
 **************************************************************************/

/*
 $Id$ 
*/


////////////////////////////////////////////////////////////////////////
//  This class is a base class for the ITS geometry version 11. It 
//  contains common/standard functions used in many places in defining 
//  the ITS geometry, version 11. Large posions of the ITS geometry, 
//  version 11, should be derived from this class so as to make maximum 
//  use of these common functions. This class also defines the proper 
//  conversion valuse such, to cm and degrees, such that the most usefull 
//  units, those used in the Engineering drawings, can be used.
////////////////////////////////////////////////////////////////////////


#include <Riostream.h>
#include <TArc.h>
#include <TLine.h>
#include <TArrow.h>
#include <TCanvas.h>
#include <TText.h>

#include <TGeoPcon.h>
#include <TGeoCone.h>
#include <TGeoTube.h> // contaings TGeoTubeSeg
#include <TGeoArb8.h>
#include <TGeoElement.h>
#include <TGeoMaterial.h>
#include <TPolyMarker.h>
#include <TPolyLine.h>

#include "V11Geometry.h"

using std::endl;
using std::cout;
using std::cin;

using namespace AliceO2::ITS;

ClassImp(V11Geometry)

const Double_t V11Geometry::fgkmicron = 1.0E-4;
const Double_t V11Geometry::fgkmm = 0.10;
const Double_t V11Geometry::fgkcm = 1.00;
const Double_t V11Geometry::fgkDegree = 1.0;
const Double_t V11Geometry::fgkRadian = 180./3.14159265358979323846;
const Double_t V11Geometry::fgkgcm3 = 1.0; // assume default is g/cm^3
const Double_t V11Geometry::fgkKgm3 = 1.0E+3;// assume Kg/m^3
const Double_t V11Geometry::fgkKgdm3 = 1.0; // assume Kg/dm^3
const Double_t V11Geometry::fgkCelsius = 1.0; // Assume default is C
const Double_t V11Geometry::fgkPascal  = 1.0E-3; // Assume kPascal
const Double_t V11Geometry::fgkKPascal = 1.0;    // Asume kPascal
const Double_t V11Geometry::fgkeV      = 1.0E-9; // GeV default
const Double_t V11Geometry::fgkKeV     = 1.0e-6; // GeV default
const Double_t V11Geometry::fgkMeV     = 1.0e-3; // GeV default
const Double_t V11Geometry::fgkGeV     = 1.0;    // GeV default

void V11Geometry::IntersectLines(Double_t m, Double_t x0, Double_t y0,
				       Double_t n, Double_t x1, Double_t y1,
				       Double_t &xi, Double_t &yi)const{
    // Given the two lines, one passing by (x0,y0) with slope m and
    // the other passing by (x1,y1) with slope n, returns the coordinates
    // of the intersecting point (xi,yi)
    // Inputs:
    //    Double_t   m     The slope of the first line
    //    Double_t  x0,y0  The x and y coord. of the first point
    //    Double_t   n     The slope of the second line
    //    Double_t  x1,y1  The x and y coord. of the second point
    // Outputs:
    //    The coordinates xi and yi of the intersection point
    // Return:
    //    none.
    // Created:      14 Dec 2009  Mario Sitta

    if (TMath::Abs(m-n) < 0.000001) {
      LOG(ERROR) << "Lines are parallel: m = " << m << " n = " << n << FairLogger::endl;
      return;
    }

    xi = (y1 - n*x1 - y0 + m*x0)/(m - n);
    yi = y0 + m*(xi - x0);

    return;
}

Bool_t V11Geometry::IntersectCircle(Double_t m, Double_t x0, Double_t y0,
					  Double_t rr, Double_t xc, Double_t yc,
					  Double_t &xi1, Double_t &yi1,
					  Double_t &xi2, Double_t &yi2){
    // Given a lines  passing by (x0,y0) with slope m and a circle with
    // radius rr and center (xc,yc), returns the coordinates of the
    // intersecting points (xi1,yi1) and (xi2,yi2) (xi1 > xi2)
    // Inputs:
    //    Double_t   m     The slope of the line
    //    Double_t  x0,y0  The x and y coord. of the point
    //    Double_t   rr     The radius of the circle
    //    Double_t  xc,yc  The x and y coord. of the center of circle
    // Outputs:
    //    The coordinates xi and yi of the intersection points
    // Return:
    //    kFALSE if the line does not intercept the circle, otherwise kTRUE
    // Created:      18 Dec 2009  Mario Sitta

    Double_t p = m*x0 - y0;
    Double_t q = m*m + 1;

    p = p-m*xc+yc;

    Double_t delta = m*m*p*p - q*(p*p - rr*rr);

    if (delta < 0)
      return kFALSE;
    else {
      Double_t root = TMath::Sqrt(delta);
      xi1 = (m*p + root)/q + xc;
      xi2 = (m*p - root)/q + xc;
      yi1 = m*(xi1 - x0) + y0;
      yi2 = m*(xi2 - x0) + y0;
      return kTRUE;
    }
}

Double_t V11Geometry::Yfrom2Points(Double_t x0,Double_t y0,
                                         Double_t x1,Double_t y1,
                                         Double_t x)const{
    // Given the two points (x0,y0) and (x1,y1) and the location x, returns
    // the value y corresponding to that point x on the line defined by the
    // two points.
    // Inputs:
    //    Double_t  x0  The first x value defining the line
    //    Double_t  y0  The first y value defining the line
    //    Double_t  x1  The second x value defining the line
    //    Double_t  y1  The second y value defining the line
    //    Double_t   x  The x value for which the y value is wanted.
    // Outputs:
    //    none.
    // Return:
    //    The value y corresponding to the point x on the line defined by
    //    the two points (x0,y0) and (x1,y1).

    if(x0==x1 && y0==y1) {
        printf("Error: V11Geometry::Yfrom2Ponts The two points are "
               "the same (%e,%e) and (%e,%e)",x0,y0,x1,y1);
        return 0.0;
    } // end if
    if(x0==x1){
        printf("Warning: V11Geometry::Yfrom2Points x0=%e == x1=%e. "
               "line vertical ""returning mean y",x0,x1);
        return 0.5*(y0+y1);
    }// end if x0==x1
    Double_t m = (y0-y1)/(x0-x1);
    return m*(x-x0)+y0;
}

Double_t V11Geometry::Xfrom2Points(Double_t x0,Double_t y0,
                                         Double_t x1,Double_t y1,
                                         Double_t y)const{
    // Given the two points (x0,y0) and (x1,y1) and the location y, returns
    // the value x corresponding to that point y on the line defined by the
    // two points.
    // Inputs:
    //    Double_t  x0  The first x value defining the line
    //    Double_t  y0  The first y value defining the line
    //    Double_t  x1  The second x value defining the line
    //    Double_t  y1  The second y value defining the line
    //    Double_t   y  The y value for which the x value is wanted.
    // Outputs:
    //    none.
    // Return:
    //    The value x corresponding to the point y on the line defined by
    //    the two points (x0,y0) and (x1,y1).

    if(x0==x1 && y0==y1) {
        printf("Error: V11Geometry::Yfrom2Ponts The two points are "
               "the same (%e,%e) and (%e,%e)",x0,y0,x1,y1);
        return 0.0;
    } // end if
    if(y0==y1){
        printf("Warrning: V11Geometry::Yfrom2Points y0=%e == y1=%e. "
               "line horizontal returning mean x",y0,y1);
        return 0.5*(x0+x1);
    }// end if y0==y1
    Double_t m = (x0-x1)/(y0-y1);
    return m*(y-y0)+x0;
}

Double_t V11Geometry::RmaxFrom2Points(const TGeoPcon *p,Int_t i1,
                                            Int_t i2,Double_t z)const{
    // functions Require at parts of Volume A to be already defined.
    // Retruns the value of Rmax corresponding to point z alone the line
    // defined by the two points p.Rmax(i1),p-GetZ(i1) and p->GetRmax(i2),
    // p->GetZ(i2).
    // Inputs:
    //    TGeoPcon *p  The Polycone where the two points come from
    //    Int_t    i1  Point 1
    //    Int_t    i2  Point 2
    //    Double_t  z  The value of z for which Rmax is to be found
    // Outputs:
    //    none.
    // Return:
    //    Double_t Rmax the value corresponding to z
    Double_t d0,d1,d2,r;

    d0 = p->GetRmax(i1)-p->GetRmax(i2);// cout <<"L263: d0="<<d0<<endl;
    d1 = z-p->GetZ(i2);// cout <<"L264: d1="<<d1<<endl;
    d2 = p->GetZ(i1)-p->GetZ(i2);// cout <<"L265: d2="<<d2<<endl;
    r  = p->GetRmax(i2) + d1*d0/d2;// cout <<"L266: r="<<r<<endl;
    return r;
}

Double_t V11Geometry::RminFrom2Points(const TGeoPcon *p,Int_t i1,
                                            Int_t i2,Double_t z)const{
    // Retruns the value of Rmin corresponding to point z alone the line
    // defined by the two points p->GetRmin(i1),p->GetZ(i1) and 
    // p->GetRmin(i2),  p->GetZ(i2).
    // Inputs:
    //    TGeoPcon *p  The Polycone where the two points come from
    //    Int_t    i1  Point 1
    //    Int_t    i2  Point 2
    //    Double_t  z  The value of z for which Rmax is to be found
    // Outputs:
    //    none.
    // Return:
    //    Double_t Rmax the value corresponding to z

    return p->GetRmin(i2)+(p->GetRmin(i1)-p->GetRmin(i2))*(z-p->GetZ(i2))/
     (p->GetZ(i1)-p->GetZ(i2));
}

Double_t V11Geometry::RFrom2Points(const Double_t *p,const Double_t *az,
                                         Int_t i1,Int_t i2,Double_t z)const{
    // Retruns the value of Rmin corresponding to point z alone the line
    // defined by the two points p->GetRmin(i1),p->GetZ(i1) and 
    // p->GetRmin(i2), p->GetZ(i2).
    // Inputs:
    //    Double_t az  Array of z values
    //    Double_t  r  Array of r values
    //    Int_t    i1  First Point in arrays
    //    Int_t    i2  Second Point in arrays
    //    Double_t z   Value z at which r is to be found
    // Outputs:
    //    none.
    // Return:
    //    The value r corresponding to z and the line defined by the two points

    return p[i2]+(p[i1]-p[i2])*(z-az[i2])/(az[i1]-az[i2]);
}

Double_t V11Geometry::Zfrom2MinPoints(const TGeoPcon *p,Int_t i1,
                                            Int_t i2,Double_t r)const{
    // Retruns the value of Z corresponding to point R alone the line
    // defined by the two points p->GetRmin(i1),p->GetZ(i1) and 
    // p->GetRmin(i2),p->GetZ(i2)
    // Inputs:
    //    TGeoPcon *p  The Poly cone where the two points come from.
    //    Int_t    i1  First Point in arrays
    //    Int_t    i2  Second Point in arrays
    //    Double_t r   Value r min at which z is to be found
    // Outputs:
    //    none.
    // Return:
    //    The value z corresponding to r min and the line defined by 
    //    the two points

    return p->GetZ(i2)+(p->GetZ(i1)-p->GetZ(i2))*(r-p->GetRmin(i2))/
     (p->GetRmin(i1)-p->GetRmin(i2));
}

Double_t V11Geometry::Zfrom2MaxPoints(const TGeoPcon *p,Int_t i1,
                                            Int_t i2,Double_t r)const{
    // Retruns the value of Z corresponding to point R alone the line
    // defined by the two points p->GetRmax(i1),p->GetZ(i1) and 
    // p->GetRmax(i2),p->GetZ(i2)
    // Inputs:
    //    TGeoPcon *p  The Poly cone where the two points come from.
    //    Int_t    i1  First Point in arrays
    //    Int_t    i2  Second Point in arrays
    //    Double_t r   Value r max at which z is to be found
    // Outputs:
    //    none.
    // Return:
    //    The value z corresponding to r max and the line defined by 
    //    the two points

    return p->GetZ(i2)+(p->GetZ(i1)-p->GetZ(i2))*(r-p->GetRmax(i2))/
     (p->GetRmax(i1)-p->GetRmax(i2));
}

Double_t V11Geometry::Zfrom2Points(const Double_t *z,const Double_t *ar,
                                         Int_t i1,Int_t i2,Double_t r)const{
    // Retruns the value of z corresponding to point R alone the line
    // defined by the two points p->GetRmax(i1),p->GetZ(i1) and 
    // p->GetRmax(i2),p->GetZ(i2)
    // Inputs:
    //    Double_t  z  Array of z values
    //    Double_t ar  Array of r values
    //    Int_t    i1  First Point in arrays
    //    Int_t    i2  Second Point in arrays
    //    Double_t r   Value r at which z is to be found
    // Outputs:
    //    none.
    // Return:
    //    The value z corresponding to r and the line defined by the two points

    return z[i2]+(z[i1]-z[i2])*(r-ar[i2])/(ar[i1]-ar[i2]);
}

Double_t V11Geometry::RmaxFromZpCone(const TGeoPcon *p,int ip,
                                           Double_t tc,Double_t z,
                                           Double_t th)const{
    // General Outer Cone surface equation Rmax.
    // Intputs:
    //     TGeoPcon  *p   The poly cone where the initial point comes from
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   z   The value of z to compute Rmax from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value Rmax correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmin[ip] at the location z.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return -tantc*(z-p->GetZ(ip))+p->GetRmax(ip)+th/costc;
}

Double_t V11Geometry::RFromZpCone(const Double_t *ar,
                                        const Double_t *az,int ip,
                                        Double_t tc,Double_t z,
                                        Double_t th)const{
    // General Cone surface equation R(z).
    // Intputs:
    //     Double_t  ar   The array of R values
    //     Double_t  az   The array of Z values
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   z   The value of z to compute R from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value R correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmax[ip] at the locatin z.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return -tantc*(z-az[ip])+ar[ip]+th/costc;
}

Double_t V11Geometry::RminFromZpCone(const TGeoPcon *p,Int_t ip,
                                           Double_t tc,Double_t z,
                                           Double_t th)const{
    // General Inner Cone surface equation Rmin.
    // Intputs:
    //     TGeoPcon  *p   The poly cone where the initial point comes from
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   z   The value of z to compute Rmin from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value Rmin correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmin[ip] at the location z.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return -tantc*(z-p->GetZ(ip))+p->GetRmin(ip)+th/costc;
}

Double_t V11Geometry::ZFromRmaxpCone(const TGeoPcon *p,int ip,
                                           Double_t tc,Double_t r,
                                           Double_t th)const{
    // General Outer cone Surface equation for z.
    // Intputs:
    //     TGeoPcon  *p   The poly cone where the initial point comes from
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   r   The value of Rmax to compute z from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value Z correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmax[ip] at the location r.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return p->GetZ(ip)+(p->GetRmax(ip)+th/costc-r)/tantc;
}

Double_t V11Geometry::ZFromRmaxpCone(const Double_t *ar,
                                           const Double_t *az,int ip,
                                           Double_t tc,Double_t r,
                                           Double_t th)const{
    // General Outer cone Surface equation for z.
    // Intputs:
    //     Double_t  ar   The array of R values
    //     Double_t  az   The array of Z values
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   r   The value of Rmax to compute z from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value Z correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmax[ip] at the locatin r.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return az[ip]+(ar[ip]+th/costc-r)/tantc;
}

Double_t V11Geometry::ZFromRminpCone(const TGeoPcon *p,int ip,
                                           Double_t tc,Double_t r,
                                           Double_t th)const{
    // General Inner cone Surface equation for z.
    // Intputs:
    //     TGeoPcon  *p   The poly cone where the initial point comes from
    //     Int_t     ip   The index in p to get the point location
    //     Double_t  tc   The angle of that part of the cone is at
    //     Double_t   r   The value of Rmin to compute z from
    //     Double_t  th   The perpendicular distance the parralell line is
    //                    from the point ip.
    // Outputs:
    //     none.
    // Return:
    //     The value Z correstponding to the line at angle th, offeset by
    //     th, and the point p->GetZ/Rmin[ip] at the location r.
    Double_t tantc = TMath::Tan(tc*TMath::DegToRad());
    Double_t costc = TMath::Cos(tc*TMath::DegToRad());

    return p->GetZ(ip)+(p->GetRmin(ip)+th/costc-r)/tantc;
}

void V11Geometry::RadiusOfCurvature(Double_t rc,Double_t theta0,
                                          Double_t z0,Double_t r0,
                                          Double_t theta1,Double_t &z1,
                                          Double_t &r1)const{
    // Given a initial point z0,r0, the initial angle theta0, and the radius
    // of curvature, returns the point z1, r1 at the angle theta1. Theta
    // measured from the r axis in the clock wise direction [degrees].
    // Inputs:
    //    Double_t rc     The radius of curvature
    //    Double_t theta0 The starting angle (degrees)
    //    Double_t z0     The value of z at theta0
    //    Double_t r0     The value of r at theta0
    //    Double_t theta1 The ending angle (degrees)
    // Outputs:
    //    Double_t &z1  The value of z at theta1
    //    Double_t &r1  The value of r at theta1
    // Return:
    //    none.

    z1 = rc*(TMath::Sin(theta1*TMath::DegToRad())-TMath::Sin(theta0*TMath::DegToRad()))+z0;
    r1 = rc*(TMath::Cos(theta1*TMath::DegToRad())-TMath::Cos(theta0*TMath::DegToRad()))+r0;
    return;
}

void V11Geometry::InsidePoint(const TGeoPcon *p,Int_t i1,Int_t i2,
                                    Int_t i3,Double_t c,TGeoPcon *q,Int_t j1,
                                    Bool_t max)const{
    // Given two lines defined by the points i1, i2,i3 in the TGeoPcon 
    // class p that intersect at point p->GetZ(i2) return the point z,r 
    // that is Cthick away in the TGeoPcon class q. If points i1=i2
    // and max == kTRUE, then p->GetRmin(i1) and p->GetRmax(i2) are used.
    // if points i2=i3 and max=kTRUE then points p->GetRmax(i2) and
    // p->GetRmin(i3) are used. If i2=i3 and max=kFALSE, then p->GetRmin(i2)
    // and p->GetRmax(i3) are used.
    // Inputs:
    //    TGeoPcon  *p  Class where points i1, i2, and i3 are taken from
    //    Int_t     i1  First point in class p
    //    Int_t     i2  Second point in class p
    //    Int_t     i3  Third point in class p
    //    Double_t  c   Distance inside the outer surface/inner suface
    //                  that the point j1 is to be computed for.
    //    TGeoPcon  *q  Pointer to class for results to be put into.
    //    Int_t     j1  Point in class q where data is to be stored.
    //    Bool_t    max if kTRUE, then a Rmax value is computed,
    //                  else a Rmin valule is computed.
    // Output:
    //    TGeoPcon  *q  Pointer to class for results to be put into.
    // Return:
    //    none.
    Double_t x0,y0,x1,y1,x2,y2,x,y;

    if(max){
        c = -c; //cout <<"L394 c="<<c<<endl;
        y0 = p->GetRmax(i1);
        if(i1==i2) y0 = p->GetRmin(i1); //cout <<"L396 y0="<<y0<<endl;
        y1 = p->GetRmax(i2);  //cout <<"L397 y1="<<y1<<endl;
        y2 = p->GetRmax(i3); //cout <<"L398 y2="<<y2<<endl;
        if(i2==i3) y2 = p->GetRmin(i3); //cout <<"L399 y2="<<y2<<endl;
    }else{ // min
        y0 = p->GetRmin(i1); //cout <<"L401 y0="<<y0<<endl;
        y1 = p->GetRmin(i2); //cout <<"L402 y1="<<y1<<endl;
        y2 = p->GetRmin(i3);
        if(i2==i3) y2 = p->GetRmax(i3); //cout <<"L404 y2="<<y2<<endl;
    } // end if
    x0 = p->GetZ(i1); //cout <<"L406 x0="<<x0<<endl;
    x1 = p->GetZ(i2); //cout <<"L407 x1="<<x1<<endl;
    x2 = p->GetZ(i3); //cout <<"L408 x2="<<x2<<endl;
    //
    InsidePoint(x0,y0,x1,y1,x2,y2,c,x,y);
    q->Z(j1) = x;
    if(max) q->Rmax(j1) = y;
    else    q->Rmin(j1) = y;
    return;
}

void V11Geometry::InsidePoint(Double_t x0,Double_t y0,
                                    Double_t x1,Double_t y1,
                                    Double_t x2,Double_t y2,Double_t c,
                                    Double_t &x,Double_t &y)const{
    // Given two intersecting lines defined by the points (x0,y0), (x1,y1) and
    // (x1,y1), (x2,y2) {intersecting at (x1,y1)} the point (x,y) a distance
    // c away is returned such that two lines a distance c away from the
    // lines defined above intersect at (x,y).
    // Inputs:
    //    Double_t  x0 X point on the first intersecting sets of lines
    //    Double_t  y0 Y point on the first intersecting sets of lines
    //    Double_t  x1 X point on the first/second intersecting sets of lines
    //    Double_t  y1 Y point on the first/second intersecting sets of lines
    //    Double_t  x2 X point on the second intersecting sets of lines
    //    Double_t  y2 Y point on the second intersecting sets of lines
    //    Double_t  c  Distance the two sets of lines are from each other
    // Output:
    //    Double_t  x  X point for the intersecting sets of parellel lines
    //    Double_t  y  Y point for the intersecting sets of parellel lines
    // Return:
    //    none.
    Double_t dx01,dx12,dy01,dy12,r01,r12,m;

    //printf("InsidePoint: x0=% #12.7g y0=% #12.7g x1=% #12.7g y1=% #12.7g "
    //       "x2=% #12.7g y2=% #12.7g c=% #12.7g ",x0,y0,x1,y2,x2,y2,c);
    dx01 = x0-x1; //cout <<"L410 dx01="<<dx01<<endl;
    dx12 = x1-x2; //cout <<"L411 dx12="<<dx12<<endl;
    dy01 = y0-y1; //cout <<"L412 dy01="<<dy01<<endl;
    dy12 = y1-y2; //cout <<"L413 dy12="<<dy12<<endl;
    r01  = TMath::Sqrt(dy01*dy01+dx01*dx01); //cout <<"L414 r01="<<r01<<endl;
    r12  = TMath::Sqrt(dy12*dy12+dx12*dx12); //cout <<"L415 r12="<<r12<<endl;
    m = dx12*dy01-dy12*dx01;
    if(m*m<DBL_EPSILON){ // m == n
        if(dy01==0.0){ // line are =
            x = x1+c; //cout <<"L419 x="<<x<<endl;
            y = y1; //cout <<"L420 y="<<y<<endl;
            //printf("dy01==0.0 x=% #12.7g y=% #12.7g\n",x,y);
            return;
        }else if(dx01==0.0){
            x = x1;
            y = y1+c;
            //printf("dx01==0.0 x=% #12.7g y=% #12.7g\n",x,y);
            return;
        }else{ // dx01!=0 and dy01 !=0.
            x = x1-0.5*c*r01/dy01; //cout <<"L434 x="<<x<<endl;
            y = y1+0.5*c*r01/dx01; //cout <<"L435 y="<<y<<endl;
            //printf("m*m<DBL_E x=% #12.7g y=% #12.7g\n",x,y);
        } // end if
        return;
    } //
    x = x1+c*(dx12*r01-dx01*r12)/m; //cout <<"L442 x="<<x<<endl;
    y = y1+c*(dy12*r01-dy01*r12)/m; //cout <<"L443 y="<<y<<endl;
    //printf("          x=% #12.7g y=% #12.7g\n",x,y);
    //cout <<"=============================================="<<endl;
    return;
}

void V11Geometry:: PrintArb8(const TGeoArb8 *a)const{
    // Prints out the content of the TGeoArb8. Usefull for debugging.
    // Inputs:
    //   TGeoArb8 *a
    // Outputs:
    //   none.
    // Return:
    //   none.

    if(!GetDebug()) return;
    printf("%s",a->GetName());
    a->InspectShape();
    return;
}

void V11Geometry:: PrintPcon(const TGeoPcon *a)const{
    // Prints out the content of the TGeoPcon. Usefull for debugging.
    // Inputs:
    //   TGeoPcon *a
    // Outputs:
    //   none.
    // Return:
    //   none.
  
    if(!GetDebug()) return;
    cout << a->GetName() << ": N=" << a->GetNz() << " Phi1=" << a->GetPhi1()
         << ", Dphi=" << a->GetDphi() << endl;
    cout << "i\t   Z   \t  Rmin \t  Rmax" << endl;
    for(Int_t iii=0;iii<a->GetNz();iii++){
        cout << iii << "\t" << a->GetZ(iii) << "\t" << a->GetRmin(iii)
             << "\t" << a->GetRmax(iii) << endl;
    } // end for iii
    return;
}

void V11Geometry::PrintTube(const TGeoTube *a)const{
    // Prints out the content of the TGeoTube. Usefull for debugging.
    // Inputs:
    //   TGeoTube *a
    // Outputs:
    //   none.
    // Return:
    //   none.

    if(!GetDebug()) return;
    cout << a->GetName() <<": Rmin="<<a->GetRmin()
         <<" Rmax=" <<a->GetRmax()<<" Dz="<<a->GetDz()<<endl;
    return;
}

void V11Geometry::PrintTubeSeg(const TGeoTubeSeg *a)const{
    // Prints out the content of the TGeoTubeSeg. Usefull for debugging.
    // Inputs:
    //   TGeoTubeSeg *a
    // Outputs:
    //   none.
    // Return:
    //   none.

    if(!GetDebug()) return;
    cout << a->GetName() <<": Phi1="<<a->GetPhi1()<<
        " Phi2="<<a->GetPhi2()<<" Rmin="<<a->GetRmin()
         <<" Rmax=" <<a->GetRmax()<<" Dz="<<a->GetDz()<<endl;
    return;
}

void V11Geometry::PrintConeSeg(const TGeoConeSeg *a)const{
    // Prints out the content of the TGeoConeSeg. Usefull for debugging.
    // Inputs:
    //   TGeoConeSeg *a
    // Outputs:
    //   none.
    // Return:
    //   none.

    if(!GetDebug()) return;
    cout << a->GetName() <<": Phi1="<<a->GetPhi1()<<
        " Phi2="<<a->GetPhi2()<<" Rmin1="<<a->GetRmin1()
         <<" Rmax1=" <<a->GetRmax1()<<" Rmin2="<<a->GetRmin2()
         <<" Rmax2=" <<a->GetRmax2()<<" Dz="<<a->GetDz()<<endl;
    return;
}

void V11Geometry::PrintBBox(const TGeoBBox *a)const{
    // Prints out the content of the TGeoBBox. Usefull for debugging.
    // Inputs:
    //   TGeoBBox *a
    // Outputs:
    //   none.
    // Return:
    //   none.

    if(!GetDebug()) return;
    cout << a->GetName() <<": Dx="<<a->GetDX()<<
        " Dy="<<a->GetDY()<<" Dz="<<a->GetDZ() <<endl;
    return;
}

void V11Geometry::CreateDefaultMaterials(){
    // Create ITS materials
    // Defined media here should correspond to the one defined in galice.cuts
    // File which is red in (AliMC*) fMCApp::Init() { ReadTransPar(); }
    // Inputs:
    //    none.
    // Outputs:
    //   none.
    // Return:
    //   none.
    Int_t i;
    Double_t w;

    // Define some elements
    TGeoElement *itsH  = new TGeoElement("ITS_H","Hydrogen",1,1.00794);
    TGeoElement *itsHe = new TGeoElement("ITS_He","Helium",2,4.002602);
    TGeoElement *itsC  = new TGeoElement("ITS_C","Carbon",6,12.0107);
    TGeoElement *itsN  = new TGeoElement("ITS_N","Nitrogen",7,14.0067);
    TGeoElement *itsO  = new TGeoElement("ITS_O","Oxygen",8,15.994);
    TGeoElement *itsF  = new TGeoElement("ITS_F","Florine",9,18.9984032);
    TGeoElement *itsNe = new TGeoElement("ITS_Ne","Neon",10,20.1797);
    TGeoElement *itsMg = new TGeoElement("ITS_Mg","Magnesium",12,24.3050);
    TGeoElement *itsAl = new TGeoElement("ITS_Al","Aluminum",13,26981538);
    TGeoElement *itsSi = new TGeoElement("ITS_Si","Silicon",14,28.0855);
    TGeoElement *itsP  = new TGeoElement("ITS_P" ,"Phosphorous",15,30.973761);
    TGeoElement *itsS  = new TGeoElement("ITS_S" ,"Sulfur",16,32.065);
    TGeoElement *itsAr = new TGeoElement("ITS_Ar","Argon",18,39.948);
    TGeoElement *itsTi = new TGeoElement("ITS_Ti","Titanium",22,47.867);
    TGeoElement *itsCr = new TGeoElement("ITS_Cr","Chromium",24,51.9961);
    TGeoElement *itsMn = new TGeoElement("ITS_Mn","Manganese",25,54.938049);
    TGeoElement *itsFe = new TGeoElement("ITS_Fe","Iron",26,55.845);
    TGeoElement *itsCo = new TGeoElement("ITS_Co","Cobalt",27,58.933200);
    TGeoElement *itsNi = new TGeoElement("ITS_Ni","Nickrl",28,56.6930);
    TGeoElement *itsCu = new TGeoElement("ITS_Cu","Copper",29,63.546);
    TGeoElement *itsZn = new TGeoElement("ITS_Zn","Zinc",30,65.39);
    TGeoElement *itsKr = new TGeoElement("ITS_Kr","Krypton",36,83.80);
    TGeoElement *itsMo = new TGeoElement("ITS_Mo","Molylibdium",42,95.94);
    TGeoElement *itsXe = new TGeoElement("ITS_Xe","Zeon",54,131.293);

    // Start with the Materials since for any one material there
    // can be defined more than one Medium.
    // Air, dry. at 15degree C, 101325Pa at sea-level, % by volume
    // (% by weight). Density is 351 Kg/m^3
    // N2 78.084% (75.47%), O2 20.9476% (23.20%), Ar 0.934 (1.28%)%,
    // C02 0.0314% (0.0590%), Ne 0.001818% (0.0012%, CH4 0.002% (),
    // He 0.000524% (0.00007%), Kr 0.000114% (0.0003%), H2 0.00005% (3.5E-6%), 
    // Xe 0.0000087% (0.00004 %), H2O 0.0% (dry) + trace amounts at the ppm
    // levels.
    TGeoMixture *itsAir = new TGeoMixture("ITS_Air",9);
    w = 75.47E-2;
    itsAir->AddElement(itsN,w);// Nitorgen, atomic
    w = 23.29E-2 + // O2
         5.90E-4 * 2.*15.994/(12.0107+2.*15.994);// CO2.
    itsAir->AddElement(itsO,w);// Oxygen, atomic
    w = 1.28E-2;
    itsAir->AddElement(itsAr,w);// Argon, atomic
    w =  5.90E-4*12.0107/(12.0107+2.*15.994)+  // CO2
         2.0E-5 *12.0107/(12.0107+4.* 1.00794);  // CH4
    itsAir->AddElement(itsC,w);// Carbon, atomic
    w = 1.818E-5;
    itsAir->AddElement(itsNe,w);// Ne, atomic
    w = 3.5E-8;
    itsAir->AddElement(itsHe,w);// Helium, atomic
    w = 7.0E-7;
    itsAir->AddElement(itsKr,w);// Krypton, atomic
    w = 3.0E-6;
    itsAir->AddElement(itsH,w);// Hydrogen, atomic
    w = 4.0E-7;
    itsAir->AddElement(itsXe,w);// Xenon, atomic
    itsAir->SetDensity(351.0*fgkKgm3); //
    itsAir->SetPressure(101325*fgkPascal);
    itsAir->SetTemperature(15.0*fgkCelsius);
    itsAir->SetState(TGeoMaterial::kMatStateGas);
    //
    // Silicone
    TGeoMaterial *itsSiDet = new TGeoMaterial("ITS_Si",itsSi,2.33*fgkgcm3);
    itsSiDet->SetTemperature(15.0*fgkCelsius);
    itsSiDet->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Epoxy C18 H19 O3
    TGeoMixture *itsEpoxy = new TGeoMixture("ITS_Epoxy",3);
    itsEpoxy->AddElement(itsC,18);
    itsEpoxy->AddElement(itsH,19);
    itsEpoxy->AddElement(itsO,3);
    itsEpoxy->SetDensity(1.8*fgkgcm3);
    itsEpoxy->SetTemperature(15.0*fgkCelsius);
    itsEpoxy->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Carbon Fiber, M55J, 60% fiber by volume. Fiber density
    // 1.91 g/cm^3. See ToryaCA M55J data sheet.
    //Begin_Html
    /*
       <A HREF="http://torayusa.com/cfa/pdfs/M55JDataSheet.pdf"> Data Sheet
       </A>
    */
    //End_Html
    TGeoMixture *itsCarbonFiber = new TGeoMixture("ITS_CarbonFiber-M55J",4);
    // Assume that the epoxy fill in the space between the fibers and so
    // no change in the total volume. To compute w, assume 1cm^3 total 
    // volume.
    w = 1.91/(1.91+(1.-.60)*itsEpoxy->GetDensity());
    itsCarbonFiber->AddElement(itsC,w);
    w = (1.-.60)*itsEpoxy->GetDensity()/(1.91+(1.-.06)*itsEpoxy->GetDensity());
    for(i=0;i<itsEpoxy->GetNelements();i++)
        itsCarbonFiber->AddElement(itsEpoxy->GetElement(i),
                                   itsEpoxy->GetWmixt()[i]*w);
    itsCarbonFiber->SetDensity((1.91+(1.-.60)*itsEpoxy->GetDensity())*fgkgcm3);
    itsCarbonFiber->SetTemperature(22.0*fgkCelsius);
    itsCarbonFiber->SetState(TGeoMaterial::kMatStateSolid);
    //
    // 
    //
    // Rohacell 51A  millable foam product.
    // C9 H13 N1 O2  52Kg/m^3
    // Elemental composition, Private comunications with
    // Bjorn S. Nilsen
    //Begin_Html
    /*
      <A HREF="http://www.rohacell.com/en/performanceplastics8344.html">
       Rohacell-A see Properties
       </A>
     */
    //End_Html
    TGeoMixture *itsFoam = new TGeoMixture("ITS_Foam",4);
    itsFoam->AddElement(itsC,9);
    itsFoam->AddElement(itsH,13);
    itsFoam->AddElement(itsN,1);
    itsFoam->AddElement(itsO,2);
    itsFoam->SetTitle("Rohacell 51 A");
    itsFoam->SetDensity(52.*fgkKgm3);
    itsFoam->SetTemperature(22.0*fgkCelsius);
    itsFoam->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Kapton % by weight, H 2.6362, C69.1133, N 7.3270, O 20.0235
    // Density 1.42 g/cm^3
    //Begin_Html
    /*
        <A HREF="http://www2.dupont.com/Kapton/en_US/assets/downloads/pdf/summaryofprop.pdf">
        Kapton. also see </A>
        <A HREF="http://physics.nist.gov/cgi-bin/Star/compos.pl?matno=179">
        </A>
     */
    //End_Html
    TGeoMixture *itsKapton = new TGeoMixture("ITS_Kapton",4);
    itsKapton->AddElement(itsH,0.026362);
    itsKapton->AddElement(itsC,0.691133);
    itsKapton->AddElement(itsN,0.073270);
    itsKapton->AddElement(itsO,0.200235);
    itsKapton->SetTitle("Kapton ribon and cable base");
    itsKapton->SetDensity(1.42*fgkgcm3);
    itsKapton->SetTemperature(22.0*fgkCelsius);
    itsKapton->SetState(TGeoMaterial::kMatStateSolid);
    //
    // UPILEX-S C16 H6 O4 N2 polymer (a Kapton like material)
    // Density 1.47 g/cm^3
    //Begin_Html
    /*
        <A HREF="http://northamerica.ube.com/page.php?pageid=9">
        UPILEX-S. also see </A>
        <A HREF="http://northamerica.ube.com/page.php?pageid=81">
        </A>
     */
    //End_Html
    TGeoMixture *itsUpilex = new TGeoMixture("ITS_Upilex",4);
    itsUpilex->AddElement(itsC,16);
    itsUpilex->AddElement(itsH,6);
    itsUpilex->AddElement(itsN,2);
    itsUpilex->AddElement(itsO,4);
    itsUpilex->SetTitle("Upilex ribon, cable, and pcb base");
    itsUpilex->SetDensity(1.47*fgkgcm3);
    itsUpilex->SetTemperature(22.0*fgkCelsius);
    itsUpilex->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Aluminum 6061 (Al used by US groups)
    // % by weight, Cr 0.04-0.35 range [0.0375 nominal value used]
    // Cu 0.15-0.4 [0.275], Fe Max 0.7 [0.35], Mg 0.8-1.2 [1.0],
    // Mn Max 0.15 [0.075] Si 0.4-0.8 [0.6], Ti Max 0.15 [0.075],
    // Zn Max 0.25 [0.125], Rest Al [97.4625]. Density 2.7 g/cm^3
    //Begin_Html
    /*
      <A HREG="http://www.matweb.com/SpecificMaterial.asp?bassnum=MA6016&group=General">
      Aluminum 6061 specifications
      </A>
     */
    //End_Html
    TGeoMixture *itsAl6061 = new TGeoMixture("ITS_Al6061",9);
    itsAl6061->AddElement(itsCr,0.000375);
    itsAl6061->AddElement(itsCu,0.00275);
    itsAl6061->AddElement(itsFe,0.0035);
    itsAl6061->AddElement(itsMg,0.01);
    itsAl6061->AddElement(itsMn,0.00075);
    itsAl6061->AddElement(itsSi,0.006);
    itsAl6061->AddElement(itsTi,0.00075);
    itsAl6061->AddElement(itsZn,0.00125);
    itsAl6061->AddElement(itsAl,0.974625);
    itsAl6061->SetTitle("Aluminum Alloy 6061");
    itsAl6061->SetDensity(2.7*fgkgcm3);
    itsAl6061->SetTemperature(22.0*fgkCelsius);
    itsAl6061->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Aluminum 7075  (Al used by Italian groups)
    // % by weight, Cr 0.18-0.28 range [0.23 nominal value used]
    // Cu 1.2-2.0 [1.6], Fe Max 0.5 [0.25], Mg 2.1-2.9 [2.5],
    // Mn Max 0.3 [0.125] Si Max 0.4 [0.2], Ti Max 0.2 [0.1],
    // Zn 5.1-6.1 [5.6], Rest Al [89.395]. Density 2.81 g/cm^3
    //Begin_Html
    /*
      <A HREG="http://asm.matweb.com/search/SpecificMaterial.asp?bassnum=MA7075T6">
      Aluminum 7075 specifications
      </A>
     */
    //End_Html
    TGeoMixture *itsAl7075 = new TGeoMixture("ITS_Al7075",9);
    itsAl7075->AddElement(itsCr,0.0023);
    itsAl7075->AddElement(itsCu,0.016);
    itsAl7075->AddElement(itsFe,0.0025);
    itsAl7075->AddElement(itsMg,0.025);
    itsAl7075->AddElement(itsMn,0.00125);
    itsAl7075->AddElement(itsSi,0.002);
    itsAl7075->AddElement(itsTi,0.001);
    itsAl7075->AddElement(itsZn,0.056);
    itsAl7075->AddElement(itsAl,0.89395);
    itsAl7075->SetTitle("Aluminum Alloy 7075");
    itsAl7075->SetDensity(2.81*fgkgcm3);
    itsAl7075->SetTemperature(22.0*fgkCelsius);
    itsAl7075->SetState(TGeoMaterial::kMatStateSolid);
    //
    // "Ruby" spheres, Al2 O3
    // "Ruby" Sphere posts, Ryton R-4 04
    //Begin_Html
    /*
      <A HREF="">
      Ruby Sphere Posts
      </A>
     */
    //End_Html
    TGeoMixture *itsRuby = new TGeoMixture("ITS_RubySphere",2);
    itsRuby->AddElement(itsAl,2);
    itsRuby->AddElement(itsO,3);
    itsRuby->SetTitle("Ruby reference sphere");
    itsRuby->SetDensity(2.81*fgkgcm3);
    itsRuby->SetTemperature(22.0*fgkCelsius);
    itsRuby->SetState(TGeoMaterial::kMatStateSolid);
    //
    //
    // Inox, AISI 304L, compoistion % by weight (assumed) 
    // C Max 0.03 [0.015], Mn Max 2.00 [1.00], Si Max 1.00 [0.50]
    // P Max 0.045 [0.0225], S Max 0.03 [0.015], Ni 8.0-10.5 [9.25]
    // Cr 18-20 [19.], Mo 2.-2.5 [2.25], rest Fe: density 7.93 Kg/dm^3
    //Begin_Html
    /*
      <A HREF="http://www.cimap.fr/caracter.pdf">
      Stainless steal (INOX) AISI 304L composition
      </A>
     */
    //End_Html
    TGeoMixture *itsInox304L = new TGeoMixture("ITS_Inox304L",9);
    itsInox304L->AddElement(itsC,0.00015);
    itsInox304L->AddElement(itsMn,0.010);
    itsInox304L->AddElement(itsSi,0.005);
    itsInox304L->AddElement(itsP,0.000225);
    itsInox304L->AddElement(itsS,0.00015);
    itsInox304L->AddElement(itsNi,0.0925);
    itsInox304L->AddElement(itsCr,0.1900);
    itsInox304L->AddElement(itsMo,0.0225);
    itsInox304L->AddElement(itsFe,0.679475); // Rest Fe
    itsInox304L->SetTitle("ITS Stainless Steal (Inox) type AISI 304L");
    itsInox304L->SetDensity(7.93*fgkKgdm3);
    itsInox304L->SetTemperature(22.0*fgkCelsius);
    itsInox304L->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Inox, AISI 316L, composition % by weight (assumed)
    // C Max 0.03 [0.015], Mn Max 2.00 [1.00], Si Max 1.00 [0.50]
    // P Max 0.045 [0.0225], S Max 0.03 [0.015], Ni 10.0-14. [12.]
    // Cr 16-18 [17.], Mo 2-3 [2.5]: density 7.97 Kg/dm^3
    //Begin_Html
    /*
      <A HREF="http://www.cimap.fr/caracter.pdf">
      Stainless steal (INOX) AISI 316L composition
      </A>
     */
    //End_Html
    TGeoMixture *itsInox316L = new TGeoMixture("ITS_Inox316L",9);
    itsInox316L->AddElement(itsC,0.00015);
    itsInox316L->AddElement(itsMn,0.010);
    itsInox316L->AddElement(itsSi,0.005);
    itsInox316L->AddElement(itsP,0.000225);
    itsInox316L->AddElement(itsS,0.00015);
    itsInox316L->AddElement(itsNi,0.12);
    itsInox316L->AddElement(itsCr,0.17);
    itsInox316L->AddElement(itsMo,0.025);
    itsInox316L->AddElement(itsFe,0.66945); // Rest Fe
    itsInox316L->SetTitle("ITS Stainless Steal (Inox) type AISI 316L");
    itsInox316L->SetDensity(7.97*fgkKgdm3);
    itsInox316L->SetTemperature(22.0*fgkCelsius);
    itsInox316L->SetState(TGeoMaterial::kMatStateSolid);
    //
    // Inox, Phynox or Elgiloy AMS 5833, composition % by weight
    // C Max 0.15 [0.15], Mn Max 2.00 [2.00], Be max 0.0001 [none]
    // Ni 18. [18.], Cr 21.5 [21.5], Mo 7.5 [7.5], Co 42 [42.]: 
    // density 8.3 Kg/dm^3
    //Begin_Html
    /*
      <A HREF="http://www.freepatentsonline.com/20070032816.html">
      Compostion of Phynox or Elgiloy AMS 5833, also see
      </A>
      <A HREF="http://www.alloywire.com/phynox_alloy.html">
      under corss reference number [0024].
      </A>
     */
    //End_Html
    TGeoMixture *itsPhynox = new TGeoMixture("ITS_Phynox",7);
    itsPhynox->AddElement(itsC,0.0015);
    itsPhynox->AddElement(itsMn,0.020);
    itsPhynox->AddElement(itsNi,0.18);
    itsPhynox->AddElement(itsCr,0.215);
    itsPhynox->AddElement(itsMo,0.075);
    itsPhynox->AddElement(itsCo,0.42);
    itsPhynox->AddElement(itsFe,0.885);
    itsPhynox->SetTitle("ITS Cooling tube alloy");
    itsPhynox->SetDensity(8.3*fgkgcm3);
    itsPhynox->SetTemperature(22.0*fgkCelsius);
    itsPhynox->SetState(TGeoMaterial::kMatStateSolid);
    //
    // G10FR4
    //
    // Demineralized Water H2O SDD & SSD Cooling liquid
    TGeoMixture *itsWater = new TGeoMixture("ITS_Water",2);
    itsWater->AddElement(itsH,2);
    itsWater->AddElement(itsO,1);
    itsWater->SetTitle("ITS Cooling Water");
    itsWater->SetDensity(1.0*fgkgcm3);
    itsWater->SetTemperature(22.0*fgkCelsius);
    itsWater->SetState(TGeoMaterial::kMatStateLiquid);
    //
    // Freon SPD Cooling liquid PerFluorobuthane C4F10
    //Begin_Html
    /*
      <A HREF=" http://st-support-cooling-electronics.web.cern.ch/st-support-cooling-electronics/default.htm">
      SPD 2 phase cooling using PerFluorobuthane
      </A>
     */
    //End_Html
    TGeoMixture *itsFreon = new TGeoMixture("ITS_SPD_Freon",2);
    itsFreon->AddElement(itsC,4);
    itsFreon->AddElement(itsF,10);
    itsFreon->SetTitle("ITS SPD 2 phase Cooling freon");
    itsFreon->SetDensity(1.52*fgkgcm3);
    itsFreon->SetTemperature(22.0*fgkCelsius);
    itsFreon->SetState(TGeoMaterial::kMatStateLiquid);
    //
    //    Int_t   ifield = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Integ();
    //    Float_t fieldm = ((AliMagF*)TGeoGlobalMagField::Instance()->GetField())->Max();

    //    Float_t tmaxfd = 0.1;//  1.0;//  Degree
    //    Float_t stemax = 1.0;//  cm
    //   Float_t deemax = 0.1;// 30.0;// Fraction of particle's energy 0<deemax<=1
    //    Float_t epsil  = 1.0E-4;//  1.0;  cm
    //    Float_t stmin  = 0.0; // cm "Default value used"

    //    Float_t tmaxfdSi = 0.1; // .10000E+01; // Degree
    //   Float_t stemaxSi = 0.0075; //  .10000E+01; // cm
    //   Float_t deemaxSi = 0.1; // Fraction of particle's energy 0<deemax<=1
    //    Float_t epsilSi  = 1.0E-4;// .10000E+01;
    /*
    Float_t stminSi  = 0.0; // cm "Default value used"

    Float_t tmaxfdAir = 0.1; // .10000E+01; // Degree
    Float_t stemaxAir = .10000E+01; // cm
    Float_t deemaxAir = 0.1; // 0.30000E-02; // Fraction of particle's energy 0<deemax<=1
    Float_t epsilAir  = 1.0E-4;// .10000E+01;
    Float_t stminAir  = 0.0; // cm "Default value used"

    Float_t tmaxfdServ = 1.0; // 10.0; // Degree
    Float_t stemaxServ = 1.0; // 0.01; // cm
    Float_t deemaxServ = 0.5; // 0.1; // Fraction of particle's energy 0<deemax<=1
    Float_t epsilServ  = 1.0E-3; // 0.003; // cm
    Float_t stminServ  = 0.0; //0.003; // cm "Default value used"

    // Freon PerFluorobuthane C4F10 see 
    // http://st-support-cooling-electronics.web.cern.ch/
    //        st-support-cooling-electronics/default.htm
    Float_t afre[2]  = { 12.011,18.9984032 };
    Float_t zfre[2]  = { 6., 9. };
    Float_t wfre[2]  = { 4.,10. };
    Float_t densfre  = 1.52;

    //CM55J
    Float_t aCM55J[4]={12.0107,14.0067,15.9994,1.00794};
    Float_t zCM55J[4]={6.,7.,8.,1.};
    Float_t wCM55J[4]={0.908508078,0.010387573,0.055957585,0.025146765};
    Float_t dCM55J = 1.63;

    //ALCM55J
    Float_t aALCM55J[5]={12.0107,14.0067,15.9994,1.00794,26.981538};
    Float_t zALCM55J[5]={6.,7.,8.,1.,13.};
    Float_t wALCM55J[5]={0.817657902,0.0093488157,0.0503618265,0.0226320885,0.1};
    Float_t dALCM55J = 1.9866;

    //Si Chips
    Float_t aSICHIP[6]={12.0107,14.0067,15.9994,1.00794,28.0855,107.8682};
    Float_t zSICHIP[6]={6.,7.,8.,1.,14., 47.};
    Float_t wSICHIP[6]={0.039730642,0.001396798,0.01169634,
 			0.004367771,0.844665,0.09814344903};
    Float_t dSICHIP = 2.36436;

    //Inox
    Float_t aINOX[9]={12.0107,54.9380, 28.0855,30.9738,32.066,
 		      58.6928,55.9961,95.94,55.845};
    Float_t zINOX[9]={6.,25.,14.,15.,16., 28.,24.,42.,26.};
    Float_t wINOX[9]={0.0003,0.02,0.01,0.00045,0.0003,0.12,0.17,0.025,0.654};
    Float_t dINOX = 8.03;

    //SDD HV microcable
    Float_t aHVm[5]={12.0107,1.00794,14.0067,15.9994,26.981538};
    Float_t zHVm[5]={6.,1.,7.,8.,13.};
    Float_t wHVm[5]={0.520088819984,0.01983871336,0.0551367996,0.157399667056, 0.247536};
    Float_t dHVm = 1.6087;

    //SDD LV+signal cable
    Float_t aLVm[5]={12.0107,1.00794,14.0067,15.9994,26.981538};
    Float_t zLVm[5]={6.,1.,7.,8.,13.};
    Float_t wLVm[5]={0.21722436468,0.0082859922,0.023028867,0.06574077612, 0.68572};
    Float_t dLVm = 2.1035;

    //SDD hybrid microcab
    Float_t aHLVm[5]={12.0107,1.00794,14.0067,15.9994,26.981538};
    Float_t zHLVm[5]={6.,1.,7.,8.,13.};
    Float_t wHLVm[5]={0.24281879711,0.00926228815,0.02574224025,0.07348667449, 0.64869};
    Float_t dHLVm = 2.0502;

    //SDD anode microcab
    Float_t aALVm[5]={12.0107,1.00794,14.0067,15.9994,26.981538};
    Float_t zALVm[5]={6.,1.,7.,8.,13.};
    Float_t wALVm[5]={0.392653705471,0.0128595919215,
 		      0.041626868025,0.118832707289, 0.431909};
    Float_t dALVm = 2.0502;

    //X7R capacitors
    Float_t aX7R[7]={137.327,47.867,15.9994,58.6928,63.5460,118.710,207.2};
    Float_t zX7R[7]={56.,22.,8.,28.,29.,50.,82.};
    Float_t wX7R[7]={0.251639432,0.084755042,0.085975822,
 		     0.038244751,0.009471271,0.321736471,0.2081768};
    Float_t dX7R = 7.14567;

    // AIR
    Float_t aAir[4]={12.0107,14.0067,15.9994,39.948};
    Float_t zAir[4]={6.,7.,8.,18.};
    Float_t wAir[4]={0.000124,0.755267,0.231781,0.012827};
    Float_t dAir = 1.20479E-3;

    // Water
    Float_t aWater[2]={1.00794,15.9994};
    Float_t zWater[2]={1.,8.};
    Float_t wWater[2]={0.111894,0.888106};
    Float_t dWater   = 1.0;

    // CERAMICS
    //     94.4% Al2O3 , 2.8% SiO2 , 2.3% MnO , 0.5% Cr2O3
    Float_t acer[5]  = { 26.981539,15.9994,28.0855,54.93805,51.9961 };
    Float_t zcer[5]  = {       13.,     8.,    14.,     25.,    24. };
    Float_t wcer[5]  = {.4443408,.5213375,.0130872,.0178135,.003421};
    Float_t denscer  = 3.6;

    // G10FR4
    Float_t zG10FR4[14] = {14.00,	20.00,	13.00,	12.00,	5.00,
 			   22.00,	11.00,	19.00,	26.00,	9.00,
 			   8.00,	6.00,	7.00,	1.00};
    Float_t aG10FR4[14] = {28.0855000,40.0780000,26.9815380,24.3050000,
 			   10.8110000,47.8670000,22.9897700,39.0983000,
 			   55.8450000,18.9984000,15.9994000,12.0107000,
 			   14.0067000,1.0079400};
    Float_t wG10FR4[14] = {0.15144894,0.08147477,0.04128158,0.00904554,
 			   0.01397570,0.00287685,0.00445114,0.00498089,
 			   0.00209828,0.00420000,0.36043788,0.27529426,
 			   0.01415852,0.03427566};
    Float_t densG10FR4= 1.8;

    //--- EPOXY  --- C18 H19 O3
    Float_t aEpoxy[3] = {15.9994, 1.00794, 12.0107} ; 
    Float_t zEpoxy[3] = {     8.,      1.,      6.} ; 
    Float_t wEpoxy[3] = {     3.,     19.,     18.} ; 
    Float_t dEpoxy = 1.8 ;

    // rohacell: C9 H13 N1 O2
    Float_t arohac[4] = {12.01,  1.01, 14.010, 16.};
    Float_t zrohac[4] = { 6.,    1.,    7.,     8.};
    Float_t wrohac[4] = { 9.,   13.,    1.,     2.};
    Float_t drohac    = 0.05;

    // If he/she means stainless steel (inox) + Aluminium and Zeff=15.3383 then
    // %Al=81.6164 %inox=100-%Al
    Float_t aInAl[5] = {27., 55.847,51.9961,58.6934,28.0855 };
    Float_t zInAl[5] = {13., 26.,24.,28.,14. };
    Float_t wInAl[5] = {.816164, .131443,.0330906,.0183836,.000919182};
    Float_t dInAl    = 3.075;

    // Kapton
    Float_t aKapton[4]={1.00794,12.0107, 14.010,15.9994};
    Float_t zKapton[4]={1.,6.,7.,8.};
    Float_t wKapton[4]={0.026362,0.69113,0.07327,0.209235};
    Float_t dKapton   = 1.42;

    //SDD ruby sph.
    Float_t aAlOxide[2]  = { 26.981539,15.9994};
    Float_t zAlOxide[2]  = {       13.,     8.};
    Float_t wAlOxide[2]  = {0.4707, 0.5293};
    Float_t dAlOxide     = 3.97;
    */
}

void V11Geometry::DrawCrossSection(const TGeoPcon *p,
                            Int_t fillc,Int_t fills,
                            Int_t linec,Int_t lines,Int_t linew,
                            Int_t markc,Int_t marks,Float_t marksize)const{
    // Draws a cross sectional view of the TGeoPcon, Primarily for debugging.
    // A TCanvas should exist first.
    //  Inputs:
    //    TGeoPcon  *p  The TGeoPcon to be "drawn"
    //    Int_t  fillc  The fill color to be used
    //    Int_t  fills  The fill style to be used
    //    Int_t  linec  The line color to be used
    //    Int_t  lines  The line style to be used
    //    Int_t  linew  The line width to be used
    //    Int_t  markc  The markder color to be used
    //    Int_t  marks  The markder style to be used
    //    Float_t marksize The marker size
    // Outputs:
    //   none.
    // Return:
    //   none.
    Int_t n=0,m=0,i=0;
    Double_t *z=0,*r=0;
    TPolyMarker *pts=0;
    TPolyLine   *line=0;

    n = p->GetNz();
    if(n<=0) return;
    m = 2*n+1;
    z = new Double_t[m];
    r = new Double_t[m];

    for(i=0;i<n;i++){
        z[i] = p->GetZ(i);
        r[i] = p->GetRmax(i);
        z[i+n] = p->GetZ(n-1-i);
        r[i+n] = p->GetRmin(n-1-i);
    } //  end for i
    z[n-1] = z[0];
    r[n-1] = r[0];

    line = new TPolyLine(n,z,r);
    pts  = new TPolyMarker(n,z,r);

    line->SetFillColor(fillc);
    line->SetFillStyle(fills);
    line->SetLineColor(linec);
    line->SetLineStyle(lines);
    line->SetLineWidth(linew);
    pts->SetMarkerColor(markc);
    pts->SetMarkerStyle(marks);
    pts->SetMarkerSize(marksize);

    line->Draw("f");
    line->Draw();
    pts->Draw();

    delete[] z;
    delete[] r;

    cout<<"Hit Return to continue"<<endl;
    cin >> n;
    delete line;
    delete pts;
    return;
}

Bool_t V11Geometry::AngleOfIntersectionWithLine(Double_t x0,Double_t y0,
                                                      Double_t x1,Double_t y1,
                                                      Double_t xc,Double_t yc,
                                                      Double_t rc,Double_t &t0,
                                                      Double_t &t1)const{
    // Computes the angles, t0 and t1 corresponding to the intersection of
    // the line, defined by {x0,y0} {x1,y1}, and the circle, defined by
    // its center {xc,yc} and radius r. If the line does not intersect the
    // line, function returns kFALSE, otherwise it returns kTRUE. If the
    // line is tangent to the circle, the angles t0 and t1 will be the same.
    // Inputs:
    //   Double_t x0   X of first point defining the line
    //   Double_t y0   Y of first point defining the line
    //   Double_t x1   X of Second point defining the line
    //   Double_t y1   Y of Second point defining the line
    //   Double_t xc   X of Circle center point defining the line
    //   Double_t yc   Y of Circle center point defining the line
    //   Double_t r    radius of circle
    // Outputs:
    //   Double_t &t0  First angle where line intersects circle
    //   Double_t &t1  Second angle where line intersects circle
    // Return:
    //    kTRUE, line intersects circle, kFALSE line does not intersect circle
    //           or the line is not properly defined point {x0,y0} and {x1,y1}
    //           are the same point.
    Double_t dx,dy,cx,cy,s2,t[4];
    Double_t a0,b0,c0,a1,b1,c1,sinthp,sinthm,costhp,costhm;
    Int_t i,j;

    t0 = 400.0;
    t1 = 400.0;
    dx = x1-x0;
    dy = y1-y0;
    cx = xc-x0;
    cy = yc-y0;
    s2 = dx*dx+dy*dy;
    if(s2==0.0) return kFALSE;

    a0 = rc*rc*s2;
    if(a0==0.0) return kFALSE;
    b0 = 2.0*rc*dx*(dx*cy-cx*dy);
    c0 = dx*dx*cy*cy-2.0*dy*dx*cy*cx+cx*cx*dy*dy-rc*rc*dy*dy;
    c0 = 0.25*b0*b0/(a0*a0)-c0/a0;
    if(c0<0.0) return kFALSE;
    sinthp = -0.5*b0/a0+TMath::Sqrt(c0);
    sinthm = -0.5*b0/a0-TMath::Sqrt(c0);

    a1 = rc*rc*s2;
    if(a1==0.0) return kFALSE;
    b1 = 2.0*rc*dy*(dy*cx-dx*cy);
    c1 = dy*dy*cx*cx-2.0*dy*dx*cy*cx+dx*dx*cy*cy-rc*rc*dx*dx;
    c1 = 0.25*b1*b1/(a1*a1)-c1/a1;
    if(c1<0.0) return kFALSE;
    costhp = -0.5*b1/a1+TMath::Sqrt(c1);
    costhm = -0.5*b1/a1-TMath::Sqrt(c1);

    t[0] = t[1] = t[2] = t[3] = 400.;
    a0 = TMath::ATan2(sinthp,costhp); if(a0<0.0) a0 += 2.0*TMath::Pi();
    a1 = TMath::ATan2(sinthp,costhm); if(a1<0.0) a1 += 2.0*TMath::Pi();
    b0 = TMath::ATan2(sinthm,costhp); if(b0<0.0) b0 += 2.0*TMath::Pi();
    b1 = TMath::ATan2(sinthm,costhm); if(b1<0.0) b1 += 2.0*TMath::Pi();
    x1 = xc+rc*TMath::Cos(a0);
    y1 = yc+rc*TMath::Sin(a0);
    s2 = dx*(y1-y0)-dy*(x1-x0);
    if(s2*s2<DBL_EPSILON) t[0] = a0*TMath::RadToDeg();
    x1 = xc+rc*TMath::Cos(a1);
    y1 = yc+rc*TMath::Sin(a1);
    s2 = dx*(y1-y0)-dy*(x1-x0);
    if(s2*s2<DBL_EPSILON) t[1] = a1*TMath::RadToDeg();
    x1 = xc+rc*TMath::Cos(b0);
    y1 = yc+rc*TMath::Sin(b0);
    s2 = dx*(y1-y0)-dy*(x1-x0);
    if(s2*s2<DBL_EPSILON) t[2] = b0*TMath::RadToDeg();
    x1 = xc+rc*TMath::Cos(b1);
    y1 = yc+rc*TMath::Sin(b1);
    s2 = dx*(y1-y0)-dy*(x1-x0);
    if(s2*s2<DBL_EPSILON) t[3] = b1*TMath::RadToDeg();
    for(i=0;i<4;i++)for(j=i+1;j<4;j++){
        if(t[i]>t[j]) {t0 = t[i];t[i] = t[j];t[j] = t0;}
    } // end for i,j
    t0 = t[0];
    t1 = t[1];
    //
    return kTRUE;
}

Double_t V11Geometry::AngleForRoundedCorners0(Double_t dx,Double_t dy,
                                                    Double_t sdr)const{
    // Basic function used to determine the ending angle and starting angles
    // for rounded corners given the relative distance between the centers
    // of the circles and the difference/sum of their radii. Case 0.
    // Inputs:
    //   Double_t dx    difference in x locations of the circle centers
    //   Double_t dy    difference in y locations of the circle centers
    //   Double_t sdr   difference or sum of the circle radii
    // Outputs:
    //   none.
    // Return:
    //   the angle in Degrees
    Double_t a,b;

    b = dy*dy+dx*dx-sdr*sdr;
    if(b<0.0) Error("AngleForRoundedCorners0",
                    "dx^2(%e)+dy^2(%e)-sdr^2(%e)=b=%e<0",dx,dy,sdr,b);
    b = TMath::Sqrt(b);
    a = -sdr*dy+dx*b;
    b = -sdr*dx-dy*b;
    return TMath::ATan2(a,b)*TMath::RadToDeg();
    
}

Double_t V11Geometry::AngleForRoundedCorners1(Double_t dx,Double_t dy,
                                                    Double_t sdr)const{
    // Basic function used to determine the ending angle and starting angles
    // for rounded corners given the relative distance between the centers
    // of the circles and the difference/sum of their radii. Case 1.
    // Inputs:
    //   Double_t dx    difference in x locations of the circle centers
    //   Double_t dy    difference in y locations of the circle centers
    //   Double_t sdr   difference or sum of the circle radii
    // Outputs:
    //   none.
    // Return:
    //   the angle in Degrees
    Double_t a,b;

    b = dy*dy+dx*dx-sdr*sdr;
    if(b<0.0) Error("AngleForRoundedCorners1",
                    "dx^2(%e)+dy^2(%e)-sdr^2(%e)=b=%e<0",dx,dy,sdr,b);
    b = TMath::Sqrt(b);
    a = -sdr*dy-dx*b;
    b = -sdr*dx+dy*b;
    return TMath::ATan2(a,b)*TMath::RadToDeg();
    
}

void V11Geometry::AnglesForRoundedCorners(Double_t x0,Double_t y0,
                                                Double_t r0,Double_t x1,
                                                Double_t y1,Double_t r1,
                                                Double_t &t0,Double_t &t1)
    const{
    // Function to compute the ending angle, for arc 0, and starting angle,
    // for arc 1, such that a straight line will connect them with no
    // discontinuities.
    //Begin_Html
    /*
      <img src="picts/ITS/V11Geometry_AnglesForRoundedCorners.gif">
     */
    //End_Html
    // Inputs:
    //    Double_t x0  X Coordinate of arc 0 center.
    //    Double_t y0  Y Coordinate of arc 0 center.
    //    Double_t r0  Radius of curvature of arc 0. For signe see figure.
    //    Double_t x1  X Coordinate of arc 1 center.
    //    Double_t y1  Y Coordinate of arc 1 center.
    //    Double_t r1  Radius of curvature of arc 1. For signe see figure.
    // Outputs:
    //    Double_t t0  Ending angle of arch 0, with respect to x axis, Degrees.
    //    Double_t t1  Starting angle of arch 1, with respect to x axis, 
    //                 Degrees.
    // Return:
    //    none.
    Double_t t;

    if(r0>=0.0&&r1>=0.0) { // Inside to inside    ++
        t = AngleForRoundedCorners1(x1-x0,y1-y0,r1-r0);
        t0 = t1 = t;
        return;
    }else if(r0>=0.0&&r1<=0.0){ // Inside to Outside  +-
        r1 = -r1; // make positive
        t = AngleForRoundedCorners0(x1-x0,y1-y0,r1+r0);
        t0 = 180.0 + t;
        if(t0<0.0) t += 360.;
        if(t<0.0) t += 360.;
        t1 = t;
        return;
    }else if(r0<=0.0&&r1>=0.0){ // Outside to Inside  -+
        r0 = - r0; // make positive
        t = AngleForRoundedCorners1(x1-x0,y1-y0,r1+r0);
        t0 = 180.0 + t;
        if(t0>180.) t0 -= 360.;
        if(t >180.) t  -= 360.;
        t1 = t;
        return;
    }else if(r0<=0.0&&r1<=0.0) { // Outside to outside --
        r0 = -r0; // make positive
        r1 = -r1; // make positive
        t = AngleForRoundedCorners0(x1-x0,y1-y0,r1-r0);
        t0 = t1 = t;
        return;
    } // end if
    return;
}

void V11Geometry::MakeFigure1(Double_t x0,Double_t y0,Double_t r0,
                                    Double_t x1,Double_t y1,Double_t r1){
    // Function to create the figure discribing how the function 
    // AnglesForRoundedCorners works.
    //
    // Inputs:
    //    Double_t x0  X Coordinate of arc 0 center.
    //    Double_t y0  Y Coordinate of arc 0 center.
    //    Double_t r0  Radius of curvature of arc 0. For signe see figure.
    //    Double_t x1  X Coordinate of arc 1 center.
    //    Double_t y1  Y Coordinate of arc 1 center.
    //    Double_t r1  Radius of curvature of arc 1. For signe see figure.
    // Outputs:
    //    none.
    // Return:
    //    none.
    Double_t t0[4],t1[4],xa0[4],ya0[4],xa1[4],ya1[4],ra0[4],ra1[4];
    Double_t xmin,ymin,xmax,ymax,h;
    Int_t j;

    for(j=0;j<4;j++) {
        ra0[j] = r0; if(j%2) ra0[j] = -r0;
        ra1[j] = r1; if(j>1) ra1[j] = -r1;
        AnglesForRoundedCorners(x0,y0,ra0[j],x1,y1,ra1[j],t0[j],t1[j]);
        xa0[j] = TMath::Abs(r0)*CosD(t0[j])+x0;
        ya0[j] = TMath::Abs(r0)*SinD(t0[j])+y0;
        xa1[j] = TMath::Abs(r1)*CosD(t1[j])+x1;
        ya1[j] = TMath::Abs(r1)*SinD(t1[j])+y1;
    } // end for j
    if(r0<0.0) r0 = -r0;
    if(r1<0.0) r1 = -r1;
    xmin = TMath::Min(x0 - r0,x1-r1);
    ymin = TMath::Min(y0 - r0,y1-r1);
    xmax = TMath::Max(x0 + r0,x1+r1);
    ymax = TMath::Max(y0 + r0,y1+r1);
    for(j=1;j<4;j++) {
        xmin = TMath::Min(xmin,xa0[j]);
        xmin = TMath::Min(xmin,xa1[j]);
        ymin = TMath::Min(ymin,ya0[j]);
        ymin = TMath::Min(ymin,ya1[j]);

        xmax = TMath::Max(xmax,xa0[j]);
        xmax = TMath::Max(xmax,xa1[j]);
        ymax = TMath::Max(ymax,ya0[j]);
        ymax = TMath::Max(ymax,ya1[j]);
    } // end for j
    if(xmin<0.0) xmin *= 1.1; else xmin *= 0.9;
    if(ymin<0.0) ymin *= 1.1; else ymin *= 0.9;
    if(xmax<0.0) xmax *= 0.9; else xmax *= 1.1;
    if(ymax<0.0) ymax *= 0.9; else ymax *= 1.1;
    j = (Int_t)(500.0*(ymax-ymin)/(xmax-xmin));
    TCanvas *can = new TCanvas("V11Geometry_AnglesForRoundedCorners",
                               "Figure for V11Geometry",500,j);
    h = ymax-ymin; if(h<0) h = -h;
    can->Range(xmin,ymin,xmax,ymax);
    TArc *c0 = new TArc(x0,y0,r0);
    TArc *c1 = new TArc(x1,y1,r1);
    TLine *line[4];
    TArrow *ar0[4];
    TArrow *ar1[4];
    for(j=0;j<4;j++){
        ar0[j] = new TArrow(x0,y0,xa0[j],ya0[j]);
        ar1[j] = new TArrow(x1,y1,xa1[j],ya1[j]);
        line[j] = new TLine(xa0[j],ya0[j],xa1[j],ya1[j]);
        ar0[j]->SetLineColor(j+1);
        ar0[j]->SetArrowSize(0.1*r0/h);
        ar1[j]->SetLineColor(j+1);
        ar1[j]->SetArrowSize(0.1*r1/h);
        line[j]->SetLineColor(j+1);
    } // end for j
    c0->Draw();
    c1->Draw();
    for(j=0;j<4;j++){
        ar0[j]->Draw();
        ar1[j]->Draw();
        line[j]->Draw();
    } // end for j
    TText *t = new TText();
    t->SetTextSize(0.02);
    Char_t txt[100];
    snprintf(txt,99,"(x0=%5.2f,y0=%5.2f)",x0,y0);
    t->DrawText(x0,y0,txt);
    snprintf(txt,99,"(x1=%5.2f,y1=%5.2f)",x1,y1);
    for(j=0;j<4;j++) {
        t->SetTextColor(j+1);
        t->DrawText(x1,y1,txt);
        snprintf(txt,99,"r0=%5.2f",ra0[j]);
        t->DrawText(0.5*(x0+xa0[j]),0.5*(y0+ya0[j]),txt);
        snprintf(txt,99,"r1=%5.2f",ra1[j]);
        t->DrawText(0.5*(x1+xa1[j]),0.5*(y1+ya1[j]),txt);
    } // end for j
}
