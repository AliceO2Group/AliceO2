/// \file Module.h
/// \brief Definition of the Module class

#ifndef ALICEO2_BASE_MODULE_H_
#define ALICEO2_BASE_MODULE_H_

#include "FairModule.h"

namespace AliceO2 {
namespace Base {

/// This is the basic class for any AliceO2 detector module, whether it is
/// sensitive or not. Detector classes depend on this.
class Module : public FairModule {

public:
  Module(const char* name, const char* title, Bool_t Active = kFALSE);

  /// Default Constructor
  Module();

  /// Default Destructor
  virtual ~Module();

  // Module composition
  virtual void Material(Int_t imat, const char* name, Float_t a, Float_t z, Float_t dens,
                        Float_t radl, Float_t absl, Float_t* buf = 0, Int_t nwbuf = 0) const;

  virtual void Mixture(Int_t imat, const char* name, Float_t* a, Float_t* z, Float_t dens,
                       Int_t nlmat, Float_t* wmat) const;

  virtual void Medium(Int_t numed, const char* name, Int_t nmat, Int_t isvol, Int_t ifield,
                      Float_t fieldm, Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil,
                      Float_t stmin, Float_t* ubuf = 0, Int_t nbuf = 0) const;

  /// Define a rotation matrix. Angles are in degrees.
  /// \param nmat on output contains the number assigned to the rotation matrix
  /// \param theta1 polar angle for axis I
  /// \param phi1 azimuthal angle for axis I
  /// \param theta2 polar angle for axis II
  /// \param phi2 azimuthal angle for axis II
  /// \param theta3 polar angle for axis III
  /// \param phi3 azimuthal angle for axis III
  virtual void Matrix(Int_t& nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2,
                      Float_t theta3, Float_t phi3) const;

  static void SetDensityFactor(Float_t density)
  {
    mDensityFactor = density;
  }

  static Float_t GetDensityFactor()
  {
    return mDensityFactor;
  }

protected:
  static Float_t mDensityFactor; ///< factor that is multiplied to all material densities (ONLY for
                                 ///< systematic studies)

private:
  Module(const Module&);
  Module& operator=(const Module&);

  ClassDef(Module, 1) // Base class for ALICE Modules
};
}
}

#endif
