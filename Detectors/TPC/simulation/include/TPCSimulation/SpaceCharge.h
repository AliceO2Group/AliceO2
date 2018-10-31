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
 *   - pad granularity in r, rphi?
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
 *   - timebins or z bins?
 */

#ifndef ALICEO2_TPC_SPACECHARGE_H
#define ALICEO2_TPC_SPACECHARGE_H

#include <TMatrixT.h>

#include "AliTPCSpaceCharge3DCalc.h"
#include "DataFormatsTPC/Defs.h"

class TH3;
class TMatrixDfwd;

class AliTPCLookUpTable3DInterpolatorD;

namespace o2
{
namespace TPC
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

  /// Allocate memory for data members
  void allocateMemory();

  /// Calculate lookup tables if initial space-charge density is provided
  void init();

  /// Calculate distortion and correction lookup tables using AliTPCSpaceChargeCalc class
  void calculateLookupTables();
  /// Update distortion and correction lookup tables by current space-charge density
  /// \param eventTime time of current event
  void updateLookupTables(float eventTime);

  /// Set omega*tau and T1, T2 tensor terms in Langevin-equation solution
  /// \param omegaTau omega*tau
  /// \param t1 T1 tensor term
  /// \param t2 T2 tensor term
  void setOmegaTauT1T2(float omegaTau, float t1, float t2);
  /// Set an initial space-charge density
  /// \param hisSCDensity 3D space-charge density histogram, expected format (phi,r,z)
  void setInitialSpaceChargeDensity(TH3* hisSCDensity);
  /// Add ions to space-charge density
  /// \param zPos z position
  /// \param phiPos phi position
  /// \param rPos radial position
  /// \param nIons number of ions
  void fillSCDensity(float zPos, float phiPos, float rPos, int nIons);
  /// Propagate space-charge density along electric field by one time slice
  void propagateSpaceCharge();
  /// Drift ion along electric field by one time slice
  /// \param point 3D coordinates of the ion
  /// \return GlobalPosition3D with coordinates of drifted ion
  GlobalPosition3D driftIon(GlobalPosition3D& point);

  /// Correct electron position using correction lookup tables
  /// \param point 3D coordinates of the electron
  void correctElectron(GlobalPosition3D& point);
  /// Distort electron position using distortion lookup tables
  /// \param point 3D coordinates of the electron
  void distortElectron(GlobalPosition3D& point);

  /// Set the space-charge distortions model
  /// \param distortionType distortion type (constant or realistic)
  void setSCDistortionType(SCDistortionType distortionType) { mSCDistortionType = distortionType; }
  /// Get the space-charge distortions model
  SCDistortionType getSCDistortionType() const { return mSCDistortionType; }

 private:
  /// Convert amount of ions into charge density C/m^3
  /// \param nIons number of ions
  /// \return space-charge density (C/m^3)
  float ions2Charge(int nIons);

  static constexpr float DvDEoverv0 = 0.0025; //! v'(E) / v0 = K / (K*E0) for ions, used in dz calculation
  static const float sEzField;                //! nominal drift field

  static constexpr int MaxZSlices = 200;     //! default number of z slices (1 ms slices)
  static constexpr int MaxPhiBins = 360;     //! default number of phi bins
  static constexpr float DriftLength = 250.; //! drift length of the TPC in (cm)
  // ion mobility K = 3.0769231 cm^2/(Vs) in Ne-CO2 90-10 published by A. Deisting
  // v_drift = K * E = 3.0769231 cm^2/(Vs) * 400 V/cm = 1230.7692 cm/s
  // t_drift = 250 cm / v_drift = 203 ms
  static constexpr float IonDriftTime = 2.03e5; //! drift time of ions for one full drift (us)
  static constexpr float RadiusInner = 85.;     //! inner radius of the TPC active area
  static constexpr float RadiusOuter = 245.;    //! outer radius of the TPC active area

  const int mInterpolationOrder; ///< order for interpolation of lookup tables: 2==quadratic, >2==cubic spline

  const int mNZSlices;          ///< number of z slices used in lookup tables
  const int mNPhiBins;          ///< number of phi bins used in lookup tables
  const int mNRBins;            ///< number of r bins used in lookup tables
  const float mLengthZSlice;    ///< length of one z bin (cm)
  const float mLengthTimeSlice; ///< ion drift time for one z slice (us)
  const float mWidthPhiBin;     ///< width of one phi bin (radians)
  const float mLengthRBin;      ///< length of one r bin (cm)

  std::vector<double> mCoordZ;   ///< vector wiht coodinates of the z bins
  std::vector<double> mCoordPhi; ///< vector wiht coodinates of the phi bins
  std::vector<double> mCoordR;   ///< vector wiht coodinates of the r bins

  bool mUseInitialSCDensity;          ///< Flag for the use of an initial space-charge density at the beginning of the simulation
  bool mInitLookUpTables;             ///< Flag to indicate if lookup tables have been calculated
  float mTimeInit;                    ///< time of last update of lookup tables
  SCDistortionType mSCDistortionType; ///< Type of space-charge distortions

  AliTPCSpaceCharge3DCalc mLookUpTableCalculator; ///< object to calculate and store correction and distortion lookup tables

  /// TODO: check fastest way to order std::vectors
  std::vector<std::vector<float>> mSpaceChargeDensityA; ///< space-charge density on the A side, stored in C/m^3 (z)(phi*r), ordering: z=[0,250], ir+iphi*nRBins
  std::vector<std::vector<float>> mSpaceChargeDensityC; ///< space-charge density on the C side, stored in C/m^3 (z)(phi*r), ordering: z=[0,-250], ir+iphi*nRBins

  /// TODO: Eliminate the need for these matrices as members, they will be owned by AliTPCLookUpTable3DInterpolatorD. AliTPCLookUpTable3DInterpolatorD needs getters for the matrices and the constructor has to be modified.
  TMatrixD** mMatrixLocalIonDriftDzA;                                      ///< matrix to store local ion drift in z direction along E field on A side
  TMatrixD** mMatrixLocalIonDriftDzC;                                      ///< matrix to store local ion drift in z direction along E field on A side
  TMatrixD** mMatrixLocalIonDriftDrphiA;                                   ///< matrix to store local ion drift in rphi direction along E field on A side
  TMatrixD** mMatrixLocalIonDriftDrphiC;                                   ///< matrix to store local ion drift in rphi direction along E field on A side
  TMatrixD** mMatrixLocalIonDriftDrA;                                      ///< matrix to store local ion drift in radial direction along E field on A side
  TMatrixD** mMatrixLocalIonDriftDrC;                                      ///< matrix to store local ion drift in radial direction along E field on C side
  std::unique_ptr<AliTPCLookUpTable3DInterpolatorD> mLookUpLocalIonDriftA; ///< lookup table for local ion drift along E field on A side
  std::unique_ptr<AliTPCLookUpTable3DInterpolatorD> mLookUpLocalIonDriftC; ///< lookup table for local ion drift along E field on C side
};

} // namespace TPC
} // namespace o2

#endif // ALICEO2_TPC_SPACECHARGE_H
