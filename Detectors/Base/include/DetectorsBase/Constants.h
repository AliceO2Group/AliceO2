/// \file Constants
/// \brief General constants
/// \author ruben.shahoyan@cern.ch

#ifndef ALICEO2_BASE_CONSTANTS
#define ALICEO2_BASE_CONSTANTS

namespace o2 {
  namespace Base {
    namespace Constants {

      constexpr float kAlmost0 = 1.17549e-38;
      constexpr float kAlmost1 = 1.f-kAlmost0;
      constexpr float kVeryBig = 1.f/kAlmost0;

      constexpr float kPI     = 3.14159274101257324e+00f;
      constexpr float k2PI    = 2.f*kPI;
      constexpr float kPIHalf = 0.5f*kPI;
      constexpr float kRad2Deg = 180.f/kPI;
      constexpr float kDeg2Rad = kPI/180.f;

      constexpr int   kNSectors   = 18;
      constexpr float kSectorSpan = 360./kNSectors;

      // conversion from B(kGaus) to curvature for 1GeV pt
      constexpr float kB2C     = -0.299792458e-3;

    }
  }
}
#endif
