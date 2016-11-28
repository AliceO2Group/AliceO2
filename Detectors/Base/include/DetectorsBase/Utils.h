/// \file Utils
/// \brief General auxilliary methods
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_UTILS
#define ALICEO2_BASE_UTILS

#include "DetectorsBase/Constants.h"
#include <math.h>

namespace AliceO2 {
  namespace Base {
    namespace Utils {    

      using namespace AliceO2::Base::Constants;

      inline void  BringTo02Pi(float &phi) { 
        // ensure angle in [0:2pi] for the input in [-pi:pi] or [0:pi]
        if (phi < 0) phi += k2PI; 
      }

      inline void  BringTo02PiGen(float &phi) { 
        // ensure angle in [0:2pi] for the any input angle
        while(phi<0)    {phi += k2PI;}
        while(phi>k2PI) {phi -= k2PI;}	
      }

      inline void  BringToPMPi(float &phi) { 
        // ensure angle in [-pi:pi] for the input in [-pi:pi] or [0:pi]
        if (phi > kPI) phi -= k2PI; 
      }

      inline void  BringToPMPiGen(float &phi) { 
        // ensure angle in [-pi:pi] for any input angle
        while(phi<-kPI)   {phi += k2PI;}
        while(phi> kPI)   {phi -= k2PI;}
      }

      inline void sincosf(float ang, float& s, float &c) {
        // consider speedup for simultaneus calculation
        s = sin(ang);
        c = cos(ang);
      }

      inline void RotateZ(float *xy, float alpha) {
        // transforms vector in tracking frame alpha to global frame
        float sn,cs, x=xy[0];
        sincosf(alpha,sn,cs);
        xy[0]=x*cs - xy[1]*sn; 
        xy[1]=x*sn + xy[1]*cs;
      }

      inline int Angle2Sector(float phi) {
        // convert angle to sector ID
        int sect = (phi*kRad2Deg)/kSectorSpan;
        sect %= kNSectors;
        return (sect<0) ? sect+kNSectors-1 : sect;
      }

      inline float Sector2Angle(int sect) {
        // convert sector to its angle center
        return kSectorSpan/2.f + (sect%kNSectors)*kSectorSpan;
      }

      inline float Angle2Alpha(float phi) {
        // convert angle to its sector alpha
        return Sector2Angle(Angle2Sector(phi));
      }

      inline float BetheBlochSolid(float bg, float rho=2.33f,float kp1=0.20f,float kp2=3.00f,
          float meanI=173e-9f,float meanZA=0.49848f) {
        //
        // This is the parameterization of the Bethe-Bloch formula inspired by Geant.
        //
        // bg  - beta*gamma
        // rho - density [g/cm^3]
        // kp1 - density effect first junction point
        // kp2 - density effect second junction point
        // meanI - mean excitation energy [GeV]
        // meanZA - mean Z/A
        //
        // The default values for the kp* parameters are for silicon. 
        // The returned value is in [GeV/(g/cm^2)].
        // 
        constexpr float mK  = 0.307075e-3f; // [GeV*cm^2/g]
        constexpr float me  = 0.511e-3f;    // [GeV/c^2]
        kp1 *= 2.303f;
        kp2 *= 2.303f;
        float bg2 = bg*bg;
        float maxT= 2.f*me*bg2;    // neglecting the electron mass

        //*** Density effect
        float d2=0.; 
        const float x = log(bg);
        const float lhwI = log(28.816*1e-9*sqrtf(rho*meanZA)/meanI);
        if (x > kp2) d2 = lhwI + x - 0.5;
        else if (x > kp1) {
          double r=(kp2-x)/(kp2-kp1);
          d2 = lhwI + x - 0.5 + (0.5 - lhwI - kp1)*r*r*r;
        }	
        return mK*meanZA*(1+bg2)/bg2*(0.5*log(2*me*bg2*maxT/(meanI*meanI)) - bg2/(1+bg2) - d2);
      }


    }
  }
}

#endif
