// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file Utils
/// \brief General auxilliary methods
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_UTILS
#define ALICEO2_BASE_UTILS

#include "DetectorsBase/Constants.h"
#include <cmath>
#include <array>

namespace o2 {
  namespace Base {
    namespace Utils {

      inline void  BringTo02Pi(float &phi) {
        // ensure angle in [0:2pi] for the input in [-pi:pi] or [0:pi]
        if (phi < 0) phi += Constants::k2PI;
      }

      inline void  BringTo02PiGen(float &phi) {
        // ensure angle in [0:2pi] for the any input angle
        while(phi<0)           {phi += Constants::k2PI;}
        while(phi>Constants::k2PI) {phi -= Constants::k2PI;}
      }

      inline void  BringToPMPi(float &phi) {
        // ensure angle in [-pi:pi] for the input in [-pi:pi] or [0:pi]
        if (phi > Constants::kPI) phi -= Constants::k2PI;
      }

      inline void  BringToPMPiGen(float &phi) {
        // ensure angle in [-pi:pi] for any input angle
        while(phi<-Constants::kPI)   {phi += Constants::k2PI;}
        while(phi> Constants::kPI)   {phi -= Constants::k2PI;}
      }

      inline void sincosf(float ang, float& s, float &c) {
        // consider speedup for simultaneus calculation
        s = sinf(ang);
        c = cosf(ang);
      }

      inline void RotateZ(std::array<float,3> &xy, float alpha) {
        // transforms vector in tracking frame alpha to global frame
        float sn,cs, x=xy[0];
        sincosf(alpha,sn,cs);
        xy[0]=x*cs - xy[1]*sn;
        xy[1]=x*sn + xy[1]*cs;
      }

      inline int Angle2Sector(float phi) {
        // convert angle to sector ID, phi can be either in 0:2pi or -pi:pi convention 
        int sect = (phi*Constants::kRad2Deg)/Constants::kSectorSpanDeg;
	if (phi<0) sect += Constants::kNSectors-1;
	return sect;
      }

      inline float Sector2Angle(int sect) {
        // convert sector to its angle center, in -pi:pi convention
	float ang = Constants::kSectorSpanRad*(0.5f + sect);
	BringToPMPi(ang);
	return ang;
      }

      inline float Angle2Alpha(float phi) {
        // convert angle to its sector alpha
        return Sector2Angle(Angle2Sector(phi));
      }

      //-------------------------------------->>>
      // recursive creation of bitmask
      template<typename T>
	constexpr int bit2Mask(T v) {
	return 0x1<<v;
      }

      template<typename T, typename... Args>
	constexpr int bit2Mask(T first, Args... args) {
	return (0x1<<first) | bit2Mask(args...);
      }
      //--------------------------------------<<<
    }
  }
}

#endif
