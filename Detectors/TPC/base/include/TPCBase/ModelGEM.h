// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ModelGEM.h
/// \brief Definition for the model calculations + simulations of the GEM efficiencies
/// \author Viktor Ratza, University of Bonn, ratza@hiskp.uni-bonn.de

// ================================================================================
// How are the electron efficiencies obtained?
// ================================================================================
// Within the scope of two PhD thesis models have been derived in order to
// describe the collection as well as the extraction efficiencies for GEM
// foils. In the following you can find a brief sketch about the procedure behind this.
// Details can be found in the following two papers which emerged out of this
// research:
//   [1] Paper Jonathan,
//   [2] Paper Viktor.
//
// Simulations / Measurements [1]:
// For different GEM geometries (standard, medium, large pitch) the electric potentials
// and fieldmaps were obtained by numerical calculations in Ansys. The fieldmaps were
// then used in Garfield++ in order to simulate the drift of the charge carriers
// and the amplification processes. In order to obtain the efficiencies a fixed
// amount of electrons has been randomely distributed above the GEM for the simulations.
// The efficiencies were thereupon derived by counting where the initial electrons and
// electrons from the amplification region ended, e.g. Copper top / bottom, anode etc.
// Indeed the simulated efficiencies are in a good agreement to measurements. See [1] for
// more details.
//
// Calculations [2]:
// In oder to get an analytic understanding of the efficiencies a simplified and
// two-dimensional model has been investigated. Simplified means: No differentian
// between an inner and an outer diameter, no Polyimide layer, no gas/no diffusion
// and two-dimensional cut of the hexagonal 3D GEM structure.
// The resulting equations are in a good agreement to the results from the simulations
// in terms of limits, offsets and curves for different pitches (in case of no diffusion).
// Nevertheless differences can be found due to the simplifications in the model.
// By introducing three fit parameter (s1, s2 and s3) the calculations can be tuned
// in a way to describe the simulated datapoints. The resulting equations
// describe the efficiencies in a full region and for different GEM geometries.
//
// The results from the fitted equations are implemented in this class.

// ================================================================================
// Remarks for naming of the variables:
// ================================================================================
//
// mFitElecEffNumberHoles (in PhD/paper: N)
//   Describes the number of holes for the 2D model calculations. The number
//   of holes is given by 2N-1, i.e. mFitElecEffNumberHoles=2 refers to 3 GEM holes (one
//   central GEM hole and two GEM holes at the outside).
//
// mFitElecEffThickness (in PhD/paper: d) [unit: micrometers]
//   Thickness of the GEM foil which
//
// mFitElecEffPitch (in PhD/paper: p) [unit: micrometers]
//   Pitch of the GEM foil.
//
// mFitElecEffHoleDiameter (in PhD/paper: L) [unit: micrometers]
//   Hole diameter for the GEM hole. There is no differentiation between an inner
//   and an outer hole diameter for the 2D model calculations.
//
// mFitElecEffWidth (in PhD/paper: w) [unit: micrometers]
//   Describes the width for a unit cell (pitch) + 2x the distance to the end of the
//   GEM electrodes. This variable can be expressed in terms of the pitch and the hole
//   diameter according to w=2p-L. It is only used for internal calculations and no
//   definition is required by the user.
//
// mFitElecEffDistancePrevStage (in PhD/paper: g1) [unit: micrometers]
//   Here g1/2 describes the distance from the center of the GEM foil to the cathode
//   or the previous amplification stage.
//
// mFitElecEffDistanceNextStage (in PhD/paper: g2) [unit: micrometers]
//   Here g2/2 describes the distance from the center of the GEM foil to the anode
//   or the next amplification stage.
//
// mGeometryTuneEta1 (in PhD/paper: s1) [unitless]
//   This is a fit parameter which has been used in order to scale eta1 for the fit
//   of the calculations to the simulations.
//
// mGeometryTuneEta2 (in PhD/paper: s2) [unitless]
//   This is a fit parameter which has been used in order to scale eta2 for the fit
//   of the calculations to the simulations.
//
// mGeometryTuneDiffusion (in PhD/paper: s3) [unitless]
//   This is a fit parameter which has been used in order to tune the equations to
//   describe diffusion.

#ifndef ALICEO2_TPC_ModelGEM_H_
#define ALICEO2_TPC_ModelGEM_H_

#include <array>

namespace o2
{
namespace tpc
{

/// \class ModelGEM

class ModelGEM
{
 public:
  /// Constructor
  ModelGEM();

  /// Destructor
  ~ModelGEM() = default;

  /// Get the electron collection efficiency of the GEM for the a given field ratio
  /// \param elecFieldAbove Electric field above the GEM in kV/cm
  /// \param gemPotential GEM potential in Volts
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getElectronCollectionEfficiency(float elecFieldAbove, float gemPotential, int geom);

  /// Get the electron extraction efficiency of the GEM for the a given field ratio
  /// \param elecFieldBelow Electric field below the GEM in kV/cm
  /// \param gemPotential GEM potential in Volts
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getElectronExtractionEfficiency(float elecFieldBelow, float gemPotential, int geom);

  /// Get the absolute gain for a given GEM (multiplication inside GEM)
  /// \param gemPotential GEM potential in Volts
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getAbsoluteGain(float gemPotential, int geom);

  /// Scale the gain curves of the individual GEM stages in order to tune the model calculations.
  /// By default this factor is set to 1.0 (i.e. no scaling).
  /// \param absGainScaling Scaling factor for absolute gain curves
  void setAbsGainScalingFactor(float absGainScaling) { mAbsGainScaling = absGainScaling; };

  /// Get the single gain fluctuation
  /// \param gemPotential GEM potential in Volts
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getSingleGainFluctuation(float gemPotential, int geom);

  /// Define a 4 GEM stack for further calculations
  /// \param geometry Array with GEM geometries (possible geometries are 0 standard, 1 medium, 2 large)
  /// \param distance Array with widths between cathode/anode and GEMs (in cm)
  /// \param potential Array with GEM potentials (in Volts)
  /// \param electricField Array with electric field configuration (in kV/cm)
  void setStackProperties(const std::array<int, 4>& geometry, const std::array<float, 5>& distance, const std::array<float, 4>& potential, const std::array<float, 5>& electricField);

  /// Calculate the energy resolution for a given GEM stack (defined with setStackProperties)
  float getStackEnergyResolution();

  /// Calculate the total effective gain for a given GEM stack (defined with setStackProperties)
  float getStackEffectiveGain();

  /// Set the attachment factor to a specific value (unit 1/cm)
  /// \param attachment Attachment factor (in 1/cm)
  void setAttachment(float attachment) { mAttachment = attachment; };

 private:
  /// Geometric parameter C1 for collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC1(int geom);

  /// Geometric parameter C2 for collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC2(int geom);

  /// Geometric parameter C3 for collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC3(int geom);

  /// Geometric parameter C4 for extraction efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC4(int geom);

  /// Geometric parameter C5 for extraction efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC5(int geom);

  /// Geometric parameter C6 for extraction efficiency.
  float getParameterC6();

  /// Geometric parameter C7 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7(float eta1, float eta2, int geom);

  /// Geometric parameter C8 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8(float eta1, float eta2, int geom);

  /// Geometric parameter C9 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC9(float eta1, float eta2, int geom);

  /// Geometric parameter C7Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C8Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C9Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC9Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C7Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7BarFromX(float intXStart, float intXEnd, int geom);

  /// Geometric parameter C8Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// Integration limits same as for C7Bar
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8BarFromX(float intXStart, float intXEnd, int geom);

  /// Geometric parameter C9Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// Integration limits same as for C7Bar
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC9BarFromX(float intXStart, float intXEnd, int geom);

  /// Flux at cathode: Term in front of lambda [sum of getLambdaCathodeF2 and getLambdaCathodef2]
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getLambdaCathode(int geom);

  /// Flux at cathode: Term in front of lambda: Basic term for central GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getLambdaCathodef2(int geom);

  /// Flux at cathode: Term in front of lambda: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getLambdaCathodeF2(int n, int geom);

  /// Flux at cathode: Term in front of mu1 [sum of getMu1CathodeF2 and getMu1Cathodef2]
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu1Cathode(int geom);

  /// Flux at cathode: Term in front of mu1: Basic term for central GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu1Cathodef2(int geom);

  /// Flux at cathode: Term in front of mu1: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu1CathodeF2(int n, int geom);

  /// Flux at cathode: Term in front of mu2 [sum of getMu2CathodeF2 and getMu2Cathodef2]
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2Cathode(int geom);

  /// Flux at cathode: Term in front of mu2: Basic term for central GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2Cathodef2(int geom);

  /// Flux at cathode: Term in front of mu2: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2CathodeF2(int n, int geom);

  /// Flux C at top electrode: Term in front of (mu2-lambda) [sum of getMu2TopF2 and getMu2Topf2]
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2Top(float intXStart, float intXEnd, int geom);

  /// Flux C at top electrode: Term in front of (mu2-lambda): Basic term for central GEM hole
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  float getMu2Topf2(float intXStart, float intXEnd);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Basic term for central GEM hole: Taylor expansion at -(w+L)/4: Order 0
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2TopfTaylorTerm0(int geom);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Basic term for central GEM hole: Taylor expansion at -(w+L)/4: Order 2
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2TopfTaylorTerm2(int geom);

  /// Flux C at top electrode: Term in front of (mu2-lambda): Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  /// \param intXStart Start value for x integration
  /// \param intXEnd End value for x integration
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2TopF2(int n, float intXStart, float intXEnd, int geom);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Additional terms for outer GEM holes (2N-1 holes in total): Taylor expansion at -(w+L)/4: Order 0
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2TopFTaylorTerm0(int n, int geom);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Additional terms for outer GEM holes (2N-1 holes in total): Taylor expansion at -(w+L)/4: Order 2
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2TopFTaylorTerm2(int n, int geom);

  /// Returns the x position on the bottom electrode of the GEM where the sign flip of the electric field in y direction
  /// appears
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getIntXEndBot(float eta1, float eta2, int geom);

  /// Returns the x position on the top electrode of the GEM where the sign flip of the electric field in y direction
  /// appears
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getIntXEndTop(float eta1, float eta2, int geom);

  /// Field ratio Eta1 for the collection efficiency where the kink starts (i.e. the plateau region ends)
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getEta1Kink1(float eta2, int geom);

  /// Field ratio Eta1 for the collection efficiency where the kink ends
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getEta1Kink2(float eta2, int geom);

  /// Field ratio Eta2 for the extraction efficiency where the kink starts
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getEta2Kink1(float eta1, int geom);

  /// Field ratio Eta2 for the extraction efficiency where the kink ends
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getEta2Kink2(float eta1, int geom);

  /// Electric field (y component) at top electrode: Q term in front of (mu2-lambda):
  /// Taylor expansion: Order 0 [sum of getMu2TopfTaylorTerm0 and getMu2TopFTaylorTerm0]
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getHtop0(int geom);

  /// Electric field (y component) at top electrode: Q term in front of (mu2-lambda):
  /// Taylor expansion: Order 2 [sum of getMu2TopfTaylorTerm2 and getMu2TopFTaylorTerm2]
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getHtop2(int geom);

  /// Due to geometric symmetries the parameters C7, C8 and C9 can be calculated just like C7Bar, C8Bar
  /// and C9Bar if we "flip" the cathode and the anode. We do this by flipping the distances
  /// mFitElecEffDistancePrevStage and mFitElecEffDistanceNextStage and use the same equations as for
  /// C7Bar, C8Bar and C9Bar.
  void flipDistanceNextPrevStage();

  /// GEM geometry definitions as it was used in fits
  const int mFitElecEffNumberHoles = 200;      ///< Number of GEM holes
  const float mFitElecEffThickness = 50.0;     ///< Thickness of the GEM in micrometers
  const float mFitElecEffHoleDiameter = 60.0;  ///< Diameter of the GEM hole (there is no differentiation between inner and outer diameter) in micrometers
  float mFitElecEffDistancePrevStage = 2110.0; ///< 2*Distance from center of GEM to previous stage (i.e. cathode or GEM) in micrometers
  float mFitElecEffDistanceNextStage = 2110.0; ///< 2*Distance from center of GEM to next stage (i.e. anode or GEM) in micrometers

  const std::array<float, 3> mFitElecEffPitch; ///< Pitch of the GEM in micrometers
  const std::array<float, 3> mFitElecEffWidth; ///< 2*Pitch-Hole diameter in micrometers

  /// Field configuration as it was used in fits
  const float mFitElecEffFieldAbove = 2000.0;                                                  ///< Electric field above the GEM in Volts/cm (for extraction efficiency scans)
  const float mFitElecEffFieldBelow = 0.0;                                                     ///< Electric field below the GEM in Volts/cm (for collection efficiency scans)
  const float mFitElecEffPotentialGEM = 300.0;                                                 ///< Electric potential applied to GEM in Volts
  const float mFitElecEffFieldGEM = mFitElecEffPotentialGEM / (mFitElecEffThickness * 0.0001); ///< Electric field inside of GEM approximated as parallel plate capacitor

  /// Scaling parameters from fits
  const std::array<float, 3> mFitElecEffTuneEta1;      ///< Tuning of field ratio eta1 (also referred to as parameter s1)
  const std::array<float, 3> mFitElecEffTuneEta2;      ///< Tuning of field ratio eta2 (also referred to as parameter s2)
  const std::array<float, 3> mFitElecEffTuneDiffusion; ///< Tuning of geometric parameter C4 in order to implement diffusion (also referred to as parameter s3)

  /// Results from absolute gain simulations
  const std::array<float, 3> mFitAbsGainConstant; ///< Constant from exponential fit function
  const std::array<float, 3> mFitAbsGainSlope;    ///< Slope from exponential fit function
  float mAbsGainScaling;                          ///< We allow a scaling factor of the gain curves for tuning (by default this factor is set to 1

  /// Results from single gain fluctuation simulations and fit to distribution
  const std::array<float, 3> mFitSingleGainF0; ///< Value for f0 in single gain fluctuation distribution
  const std::array<float, 3> mFitSingleGainU0; ///< Value for U0 in single gain fluctuation distribution
  const std::array<float, 3> mFitSingleGainQ;  ///< Value for Q in single gain fluctuation distribution

  /// Some parameters are constant for a fixed GEM pitch, so we evaluate them once in the constructor
  std::array<float, 3> mParamC1;
  std::array<float, 3> mParamC2;
  std::array<float, 3> mParamC3;
  std::array<float, 3> mParamC4;
  std::array<float, 3> mParamC5;
  std::array<float, 3> mParamC6;

  /// Properties of quadruple GEM stack
  std::array<int, 4> mGeometry;        ///< Array with GEM geometries (possible geometries are 0 standard, 1 medium, 2 large)
  std::array<float, 5> mDistance;      ///< Array with widths between cathode/anode and GEMs (in cm)
  std::array<float, 4> mPotential;     ///< Array with GEM potentials (in Volts)
  std::array<float, 5> mElectricField; ///< Array with electric field configuration (in kV/cm)

  /// Total effective gain of a given stack [defined in setStackProperties()]
  float mStackEffectiveGain;

  /// Flag in order to check if getStackEnergyResolution() has been called.
  /// We use this in getStackEffectiveGain() as the value mStackEffectiveGain is only available after
  /// a call of getStackEnergyResolution().
  /// By default this is 0.
  int mStackEnergyCalculated;

  /// Electron attachment factor (in 1/cm). By default this is 0.0 1/cm.
  /// Usually this is a function of the O2 and H20 value + the strength of the electric field. A model for this is currently
  /// not implemented. However a constant value can be assumed here to allow some basic tuning.
  float mAttachment;

  const float Pi = 3.1415926;

  // Energy of incident photons (eV)
  const float PhotonEnergy = 5900.0;

  // Mean energy to create electron-ion pair in gas (here NeCO2N2, in eV)
  const float Wi = 37.3;

  // Fano factor for NeCO2N2 90-10-5 (Please check this!)
  const float Fano = 0.13;
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_ModelGEM_H_
