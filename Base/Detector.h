/// \file UpgradeSegmentationPixel.h
/// \brief Definition of the UpgradeSegmentationPixel class

#ifndef ALICEO2_BASE_DETECTOR_H_
#define ALICEO2_BASE_DETECTOR_H_

#include "FairDetector.h"

namespace AliceO2 {
namespace Base {

/// This is the basic class for any AliceO2 detector module, whether it is
/// sensitive or not. Detector classes depend on this.
class Detector : public FairDetector {

public:
  enum Model {
    test123 = 0
  };


  Detector(const char* name, Bool_t Active, Int_t DetId = 0);

  /// Default Constructor
  Detector();

  /// Default Destructor
  virtual ~Detector();

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
  /// \param theta2 polar angle for axis II
  /// \param theta3 polar angle for axis III
  /// \param phi1 azimuthal angle for axis I
  /// \param phi2 azimuthal angle for axis II
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

  /// Sets per wrapper volume parameters
  virtual void DefineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan);

  /// Books arrays for wrapper volumes
  virtual void SetNumberOfWrapperVolumes(Int_t n);

  virtual void DefineLayer(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd,
                           Int_t nmod, Double_t lthick = 0., Double_t dthick = 0.,
                           UInt_t detType = 0, Int_t buildFlag = 0);

  virtual void DefineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd,
                                Int_t nmod, Double_t width, Double_t tilt, Double_t lthick = 0.,
                                Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0);

protected:
  static Float_t mDensityFactor; ///< factor that is multiplied to all material densities (ONLY for
                                 ///< systematic studies)
private:
  Detector(const Detector&);
  Detector& operator=(const Detector&);

  ClassDef(Detector, 1) // Base class for ALICE Modules
};
}
}

#endif
