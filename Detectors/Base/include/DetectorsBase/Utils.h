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

    }
  }
}

#endif
