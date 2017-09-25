// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file UpgradeSegmentationPixel.h
/// \brief Definition of the UpgradeSegmentationPixel class

#ifndef ALICEO2_BASE_DETECTOR_H_
#define ALICEO2_BASE_DETECTOR_H_

#include <map>
#include <vector>
#include <memory>

#include "FairDetector.h"  // for FairDetector
#include "Rtypes.h"        // for Float_t, Int_t, Double_t, Detector::Class, etc
#include <cxxabi.h>
#include <typeinfo>
#include <type_traits>

namespace o2 {
namespace Base {

/// This is the basic class for any AliceO2 detector module, whether it is
/// sensitive or not. Detector classes depend on this.
class Detector : public FairDetector
{

  public:
    Detector(const char *name, Bool_t Active, Int_t DetId = 0);

    /// Default Constructor
    Detector();

    /// Default Destructor
    ~Detector() override;

    // Module composition
    void Material(Int_t imat, const char *name, Float_t a, Float_t z, Float_t dens, Float_t radl, Float_t absl,
                  Float_t *buf = nullptr, Int_t nwbuf = 0);

    void Mixture(Int_t imat, const char *name, Float_t *a, Float_t *z, Float_t dens, Int_t nlmat,
                 Float_t *wmat);

    void Medium(Int_t numed, const char *name, Int_t nmat, Int_t isvol, Int_t ifield, Float_t fieldm,
                Float_t tmaxfd, Float_t stemax, Float_t deemax, Float_t epsil, Float_t stmin, Float_t *ubuf = nullptr,
                Int_t nbuf = 0);

    /// Define a rotation matrix. angles are in degrees.
    /// \param nmat on output contains the number assigned to the rotation matrix
    /// \param theta1 polar angle for axis I
    /// \param theta2 polar angle for axis II
    /// \param theta3 polar angle for axis III
    /// \param phi1 azimuthal angle for axis I
    /// \param phi2 azimuthal angle for axis II
    /// \param phi3 azimuthal angle for axis III
    void Matrix(Int_t &nmat, Float_t theta1, Float_t phi1, Float_t theta2, Float_t phi2, Float_t theta3,
                Float_t phi3) const;

    static void setDensityFactor(Float_t density)
    {
      mDensityFactor = density;
    }

    static Float_t getDensityFactor()
    {
      return mDensityFactor;
    }

    /// Sets per wrapper volume parameters
    virtual void defineWrapperVolume(Int_t id, Double_t rmin, Double_t rmax, Double_t zspan);

    /// Books arrays for wrapper volumes
    virtual void setNumberOfWrapperVolumes(Int_t n);

    virtual void defineLayer(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd, Int_t nmod,
                             Double_t lthick = 0., Double_t dthick = 0., UInt_t detType = 0, Int_t buildFlag = 0);

    virtual void defineLayerTurbo(Int_t nlay, Double_t phi0, Double_t r, Double_t zlen, Int_t nladd, Int_t nmod,
                                  Double_t width, Double_t tilt, Double_t lthick = 0., Double_t dthick = 0.,
                                  UInt_t detType = 0, Int_t buildFlag = 0);

    // returns global material ID given a "local" material ID for this detector
    // returns -1 in case local ID not found
    int getMaterialID(int imat) const {
      auto iter = mMapMaterial.find(imat);
      if (iter != mMapMaterial.end()){
        return iter->second;
      }
      return -1;
    }

    // returns global medium ID given a "local" medium ID for this detector
    // returns -1 in case local ID not found
    int getMediumID(int imed) const {
      auto iter = mMapMedium.find(imed);
      if (iter != mMapMedium.end()){
        return iter->second;
      }
      return -1;
    }

    // fill the medium index mapping into a standard vector
    // the vector gets sized properly and will be overridden
    void getMediumIDMappingAsVector(std::vector<int>& mapping) {
      mapping.clear();
      // get the biggest mapped value (maps are sorted in keys)
      auto maxkey = mMapMedium.rbegin()->first;
      // resize mapping and initialize with -1 by default
      mapping.resize(maxkey + 1, -1);
      // fill vector with entries from map
      for (auto& p : mMapMedium) {
        mapping[p.first] = p.second;
      }
    }

    // static and reusable service function to set tracking parameters in relation to field
    // returns global integration mode (inhomogenety) for the field and the max field value
    // which is required for media creation
    static void initFieldTrackingParams(int &mode, float &maxfield);

  protected:
    Detector(const Detector &origin);

    Detector &operator=(const Detector &);


    /// Mapping of the ALICE internal material number to the one
    /// automatically assigned by geant/TGeo.
    /// This is required to easily being able to copy the geometry setup
    /// used in AliRoot
    std::map<int, int> mMapMaterial; //!< material mapping

    /// See comment for mMapMaterial
    std::map<int, int> mMapMedium;   //!< medium mapping

    static Float_t mDensityFactor; //! factor that is multiplied to all material densities (ONLY for
    // systematic studies)

    ClassDefOverride(Detector, 1) // Base class for ALICE Modules
};

/// utility function to demangle cxx type names
inline std::string demangle(const char* name)
{
  int status = -4; // some arbitrary value to eliminate compiler warnings
  std::unique_ptr<char, void (*)(void*)> res{ abi::__cxa_demangle(name, nullptr, nullptr, &status), std::free };
  return (status == 0) ? res.get() : name;
}

// a utility function to get the unmangled original (un-aliased)
// type name given a type to be used as arguments for TClonesArray constructor
// (which cannot otherwise resolve alias typenames); This function also performs also strong
// requirements check if T is compatible with a TClonesArray (needs to inherit from TObject)
// AT COMPILE TIME
template <typename T>
inline std::string getTClArrTrueTypeName()
{
  static_assert(std::is_base_of<TObject, T>::value, "type needs to inherit from TObject");
  return demangle(typeid(T).name()).c_str();
}
}
}

#endif
