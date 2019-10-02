// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file SpaceCharge.h
/// \brief Definition of the handler for the ALICE TPC space-charge distortions calculations
/// \author Ernst Hellbär, Goethe-Universität Frankfurt, ernst.hellbar@cern.ch

/*
 * TODO:
 *   - fix constants (more precise values, export into TPCBase/Constants)
 *   - replace TMath functions/constants by std:math functions and o2 constants?
 *   - granularity in r, rphi, z?
 *   - accumulate and add next slice
 *     - event based: propagate charge(ievent-1), add charge(ievent)
 *     - time based: mTime0, mEffectiveTime
 *                   addIons(eventtime, drifttime, r, phi)
 *                   time in us, 50 kHz = <one event / 20 us>
 *                   if (ev.time+dr.time-mTime0 < mLengthTimebin) => add2NextSlice
 *                   if  (mLengthTimebin < ev.time+dr.time-mTime0 < mLengthTimebin+100us) add2NextToNextSlice
 *                     - apply updated distortions to ions in NextToNextSlice when space charge is propagated; need to store exact positions (e.g. std::vector<std::vector<float>>)!
 *   - ion transport along the E field -> Jacobi matrices?
 *   - what about primary ionization?
 *   - irregular bin sizes in r and rphi
 */

#ifndef ALICEO2_TPC_SPACECHARGE_H
#define ALICEO2_TPC_SPACECHARGE_H

#include <TMatrixT.h>

#include "AliTPCSpaceCharge3DCalc.h"
#include "DataFormatsTPC/Defs.h"
#include "TPCBase/RandomRing.h"

class TH3;
class TMatrixDfwd;

class AliTPCLookUpTable3DInterpolatorD;

namespace o2
{
namespace tpc
{

class SpaceCharge
{
 public:
  /// Enumerator for setting the space-charge distortion mode
  enum SCDistortionType {
    SCDistortionsConstant = 0, // space-charge distortions constant over time
    SCDistortionsRealistic = 1 // realistic evolution of space-charge distortions over time
  };

  // Constructors
  /// Default constructor using a grid size of (129 z bins, 180 phi bins, 129 r bins)
  SpaceCharge();
  /// Constructor with grid size specified by user
  /// \param nZSlices number of grid points in z, must be (2**N)+1
  /// \param nPhiBins number of grid points in phi
  /// \param nRBins number of grid points in r, must be (2**N)+1
  SpaceCharge(int nZSlices, int nPhiBins, int nRBins);
  /// Constructor with grid size and interpolation order specified by user
  /// \param nZSlices number of grid points in z, must be (2**N)+1
  /// \param nPhiBins number of grid points in phi
  /// \param nRBins number of grid points in r, must be (2**N)+1
  /// \param interpolationOrder order used for interpolation of lookup tables
  SpaceCharge(int nZSlices, int nPhiBins, int nRBins, int interpolationOrder);

  // Destructor
  ~SpaceCharge() = default;

  /// Calculate lookup tables if initial space-charge density is provided
  void init();

  /// Calculate distortion and correction lookup tables using AliTPCSpaceChargeCalc class
  /// \return real time for the calculation of the electron lookup tables
  float calculateLookupTables();
  /// Update distortion and correction lookup tables by current space-charge density
  /// \param eventTime time of current event
  /// \return real time for the re-calculation of the electron lookup tables
  float updateLookupTables(float eventTime);

  /// Set omega*tau and T1, T2 tensor terms in Langevin-equation solution
  /// \param omegaTau omega*tau
  /// \param t1 T1 tensor term
  /// \param t2 T2 tensor term
  void setOmegaTauT1T2(float omegaTau, float t1, float t2);
  /// Set an initial space-charge density
  /// \param hisSCDensity 3D space-charge density histogram, expected format (phi,r,z) and units C / cm^3 / epsilon0
  void setInitialSpaceChargeDensity(TH3* hisSCDensity);
  /// Add primary ions to space-charge density
  /// \param r global radius
  /// \param phi global phi position
  /// \param z z position
  /// \param nIons number of ions
  void fillPrimaryIons(double r, double phi, double z, int nIons);
  /// Add charge to space-charge density
  /// \param r global radius
  /// \param phi global phi position
  /// \param z z position
  /// \param charge charge in C/cm^3/epsilon0
  void fillPrimaryCharge(double r, double phi, double z, float charge);
  /// Add ion backflow to space-charge density
  /// \param r global radius
  /// \param phi global phi position
  /// \param side A or C side
  /// \param nIons number of ions
  void fillIBFIons(double r, double phi, Side side, int nIons);
  /// Add ion backflow to space-charge density
  /// \param r global radius
  /// \param phi global phi position
  /// \param side A or C side
  /// \param charge charge in C/cm^3/epsilon0
  void fillIBFCharge(double r, double phi, Side side, float charge);
  /// Get ion drift vector along electric field
  /// \param r global radius
  /// \param phi global phi position
  /// \param z z position
  /// \param dr return drift in radial direction
  /// \param drphi return drift in azimuthal (rphi) direction
  /// \param dz return drift in z direction
  void getIonDrift(Side side, double r, double phi, double z, double& dr, double& drphi, double& dz);
  /// Propagate space-charge density along electric field by one time slice
  void propagateSpaceCharge();
  /// Convert space-charge density to distribution of ions, propagate them along the electric field and convert back to space-charge density
  void propagateIons();

  /// Correct electron position using correction lookup tables
  /// \param point 3D coordinates of the electron
  void correctElectron(GlobalPosition3D& point);
  /// Distort electron position using distortion lookup tables
  /// \param point 3D coordinates of the electron
  void distortElectron(GlobalPosition3D& point);

  /// Interpolate the space-charge density from lookup tables in mLookUpTableCalculator
  /// \param point Position at which to calculate the space-charge density
  /// \return space-charge density at given point in C/cm^3/epsilon0
  double getChargeDensity(Side side, GlobalPosition3D& point);
  /// Get the space-charge density stored in the
  /// \param iphi phi bin
  /// \param ir r bin
  /// \param iz z bin
  /// \return space-charge density in given bin in C/cm^3/epsilon0
  float getChargeDensity(Side side, int iphi, int ir, int iz);

  /// Set the space-charge distortions model
  /// \param distortionType distortion type (constant or realistic)
  void setSCDistortionType(SCDistortionType distortionType) { mSCDistortionType = distortionType; }
  /// Get the space-charge distortions model
  SCDistortionType getSCDistortionType() const { return mSCDistortionType; }

  /// Return the ion drift time for one z bin
  double getDriftTimeZSlice() const { return mDriftTimeZSlice; }
  double getVoxelSizePhi() const { return mWidthPhiBin; }
  double getVoxelSizeR() const { return mLengthRBin; }
  double getVoxelSizeZ() const { return mLengthZSlice; }
  std::vector<double> getCoordinatesPhi() const { return mCoordPhi; }
  std::vector<double> getCoordinatesR() const { return mCoordR; }
  std::vector<double> getCoordinatesZ() const { return mCoordZ; }

  void setUseIrregularLUTs(int useIrrLUTs);
  void setUseFastDistIntegration(int useFastInt);

 private:
  /// Allocate memory for data members
  void allocateMemory();

  /// Convert amount of ions into charge density C/cm^3/epsilon0
  /// \param nIons number of ions
  /// \return space-charge density (C/cm^3/epsilon0)
  float ions2Charge(int rBin, int nIons);

  static constexpr float DvDEoverv0 = 0.0025; //! v'(E) / v0 = K / (K*E0) for ions, used in dz calculation
  static const float sEzField;                //! nominal drift field

  static constexpr int MaxZSlices = 200;     //! default number of z slices (1 ms slices)
  static constexpr int MaxPhiBins = 360;     //! default number of phi bins
  static constexpr float DriftLength = 250.; //! drift length of the TPC in (cm)
  // ion mobility K = 3.0769231 cm^2/(Vs) in Ne-CO2 90-10 published by A. Deisting
  // v_drift = K * E = 3.0769231 cm^2/(Vs) * 400 V/cm = 1230.7692 cm/s
  // t_drift = 249.7 cm / v_drift = 203 ms
  static constexpr float IonDriftTime = 2.02e5; //! drift time of ions for one full drift (us)

  const int mInterpolationOrder; ///< order for interpolation of lookup tables: 2==quadratic, >2==cubic spline

  const int mNZSlices;           ///< number of z slices used in lookup tables
  const int mNPhiBins;           ///< number of phi bins used in lookup tables
  const int mNRBins;             ///< number of r bins used in lookup tables
  const double mLengthZSlice;    ///< length of one z bin (cm)
  const double mDriftTimeZSlice; ///< ion drift time for one z slice (us)
  const double mWidthPhiBin;     ///< width of one phi bin (radians)
  const double mLengthRBin;      ///< length of one r bin (cm)

  std::vector<double> mCoordZ;   ///< vector with coodinates of the z bins
  std::vector<double> mCoordPhi; ///< vector with coodinates of the phi bins
  std::vector<double> mCoordR;   ///< vector with coodinates of the r bins

  bool mUseInitialSCDensity;          ///< Flag for the use of an initial space-charge density at the beginning of the simulation
  bool mInitLookUpTables;             ///< Flag to indicate if lookup tables have been calculated
  float mTimeInit;                    ///< time of last update of lookup tables
  SCDistortionType mSCDistortionType; ///< Type of space-charge distortions

  AliTPCSpaceCharge3DCalc mLookUpTableCalculator; ///< object to calculate and store correction and distortion lookup tables

  /// TODO: What are the coordinates of the bins? They are defined in AliTPCSpaceCharge3DCalc::GetChargeDensity and are different from mCoordZ, mCoordPhi, mCoordR used for local ion drift lookup table! Use consistent convention? Lookup table instead of vector?
  std::vector<float> mSpaceChargeDensityA; ///< space-charge density on the A side, stored in C/cm^3/epsilon0, z ordering: z=[0,250], [iphi*mNRBins*mNZSlices + ir*mNZSlices + iz]
  std::vector<float> mSpaceChargeDensityC; ///< space-charge density on the C side, stored in C/cm^3/epsilon0, z ordering: z=[0,-250], [iphi*mNRBins*mNZSlices + ir*mNZSlices + iz]

  /// Ion drift vectors after time deltaT = mLengthZSlice / v_driftIon
  /// nominal E field only in z direction, distortions in r, phi, z due to space charge
  /// d = (dr, drphi, mLengthZSlice + dz)
  /// TODO: Eliminate the need for these matrices as members, they will be owned by AliTPCLookUpTable3DInterpolatorD. AliTPCLookUpTable3DInterpolatorD needs getters for the matrices and the constructor has to be modified.
  TMatrixD** mMatrixIonDriftZA;                                       ///< matrix to store ion drift in z direction along E field on A side in cm
  TMatrixD** mMatrixIonDriftZC;                                       ///< matrix to store ion drift in z direction along E field on A side in cm
  TMatrixD** mMatrixIonDriftRPhiA;                                    ///< matrix to store ion drift in rphi direction along E field on A side in cm
  TMatrixD** mMatrixIonDriftRPhiC;                                    ///< matrix to store ion drift in rphi direction along E field on A side in cm
  TMatrixD** mMatrixIonDriftRA;                                       ///< matrix to store ion drift in radial direction along E field on A side in cm
  TMatrixD** mMatrixIonDriftRC;                                       ///< matrix to store ion drift in radial direction along E field on C side in cm
  std::unique_ptr<AliTPCLookUpTable3DInterpolatorD> mLookUpIonDriftA; ///< lookup table for ion drift along E field on A side in cm
  std::unique_ptr<AliTPCLookUpTable3DInterpolatorD> mLookUpIonDriftC; ///< lookup table for ion drift along E field on C side in cm

  /// Circular random buffer containing flat random values to convert the charge density to a flat ion distribution inside the voxel
  RandomRing<> mRandomFlat;
};

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_SPACECHARGE_H
