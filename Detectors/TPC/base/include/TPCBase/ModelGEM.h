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
// Within the scope of two PhD thesis (by Viktor Ratza and Jonathan Ottnad, HISKP University
// of Bonn, 2019) models have been derived in order to
// describe the collection as well as the extraction efficiencies for GEM
// foils. In the following you can find a brief sketch about the procedure behind this.
//
// Simulations / Measurements:
// For different GEM geometries (standard, medium, large pitch) the electric potentials
// and fieldmaps were obtained by numerical calculations in Ansys. The fieldmaps were
// then used in Garfield++ in order to simulate the drift of the charge carriers
// and the amplification processes. In order to obtain the efficiencies, a fixed
// amount of electrons has been randomely distributed above the GEM for the simulations.
// The efficiencies were thereupon derived by counting where the initial electrons and
// electrons from the amplification region ended, e.g. Copper top / bottom, anode etc.
// Indeed the simulated efficiencies are in a good agreement to measurements.
//
// Calculations [2]:
// In oder to get an analytic understanding of the efficiencies, a simplified and
// two-dimensional model has been investigated. Simplified means: No differentian
// between an inner and an outer diameter, no Polyimide layer, no gas/no diffusion
// and two-dimensional cut of the hexagonal 3D GEM structure.
// The resulting equations are in a good agreement to the results from the simulations
// in terms of limits, offsets and curves for different pitches (in case of no diffusion).
// Nevertheless differences can be found due to the simplifications in the model.
// By introducing three fit parameter, the calculations can be tuned
// in a way to describe the simulated datapoints. The resulting equations
// describe the efficiencies in a full region and for different GEM pitches.
//
// The results from the fitted equations are implemented in this class for NeCO2N2 90-10-5.

#ifndef ALICEO2_TPC_ModelGEM_H_
#define ALICEO2_TPC_ModelGEM_H_

#include <array>

namespace o2
{
namespace TPC
{

/// \class ModelGEM

class ModelGEM
{
 public:
  /// Constructor
  ModelGEM();

  /// Destructor
  ~ModelGEM() = default;

  /// Get the electron collection efficiency of the GEM for the given field configuration
  /// \param elecFieldAbove Electric field above the GEM in kV/cm
  /// \param gemPotential GEM potential in volt
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getElectronCollectionEfficiency(float elecFieldAbove, float gemPotential, int geom);

  /// Get the electron extraction efficiency of the GEM for the given field configuration
  /// \param elecFieldBelow Electric field below the GEM in kV/cm
  /// \param gemPotential GEM potential in volt
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getElectronExtractionEfficiency(float elecFieldBelow, float gemPotential, int geom);

  /// Get the absolute gain (=multiplication) for a given GEM
  /// \param gemPotential GEM potential in volt
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getAbsoluteGain(float gemPotential, int geom);

  /// Scale the gain curves of the individual GEM stages in order to tune the model calculations.
  /// By default this factor is set to 1.0 (i.e. no scaling).
  /// \param absGainScaling Scaling factor for absolute gain curves
  void setAbsGainScalingFactor(float absGainScaling) { mAbsGainScaling = absGainScaling; };

  /// Get the single gain fluctuation of a GEM
  /// \param gemPotential GEM potential in volt
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getSingleGainFluctuation(float gemPotential, int geom);

  /// Define a 4 GEM stack for further calculations
  /// \param geometry Array with GEM geometries (possible geometries are 0 standard, 1 medium, 2 large)
  /// \param distance Array with distances between cathode-GEM1, GEM1-GEM2, GEM2-GEM3, GEM3-GEM4, GEM4-anode (in cm)
  /// \param potential Array with GEM potentials (in volt)
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
  /// Geometric parameter C1 for electron collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC1(int geom);

  /// Geometric parameter C2 for electron collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC2(int geom);

  /// Geometric parameter C3 for electron collection efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC3(int geom);

  /// Geometric parameter C4 for electron extraction efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC4(int geom);

  /// Geometric parameter C5 for electron extraction efficiency
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC5(int geom);

  /// Geometric parameter C6 for electron extraction efficiency.
  /// This parameter turns out to be constant.
  float getParameterC6();

  /// Geometric parameter C7 for electron extraction efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7(float eta1, float eta2, int geom);

  /// Geometric parameter C8 for electron extraction efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8(float eta1, float eta2, int geom);

  /// Geometric parameter C9 for electron extraction efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC9(float eta1, float eta2, int geom);

  /// Geometric parameter C7Bar for electron collection efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C8Bar for electron collection efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C9Bar for electron collection efficiency as function of electric field ratios
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC9Bar(float eta1, float eta2, int geom);

  /// Geometric parameter C7Bar for electron collection efficiency as function of the integration limits on the top GEM electrode
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC7BarFromX(float intXStart, float intXEnd, int geom);

  /// Geometric parameter C8Bar for electron collection efficiency as function of the integration limits on the top GEM electrode
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getParameterC8BarFromX(float intXStart, float intXEnd, int geom);

  /// Geometric parameter C9Bar for electron collection efficiency as function of the integration limits on the top GEM electrode
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
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
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getMu2Top(float intXStart, float intXEnd, int geom);

  /// Flux C at top electrode: Term in front of (mu2-lambda): Basic term for central GEM hole
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
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
  /// \param intXStart Start value for x integration in micrometers
  /// \param intXEnd End value for x integration in micrometers
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

  /// Returns the x position (in micrometers) on the bottom electrode of the GEM where the sign flip of the electric field in y direction appears
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  /// \param geom Geometry of the GEM (0 standard, 1 medium, 2 large)
  float getIntXEndBot(float eta1, float eta2, int geom);

  /// Returns the x position (in micrometers) on the top electrode of the GEM where the sign flip of the electric field in y direction appears
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

  const std::array<float, 3> mFitElecEffPitch; ///< Pitch of the GEM geometries (standard, medium, large) in micrometers
  const std::array<float, 3> mFitElecEffWidth; ///< 2*Pitch-HoleDiameter of the GEM geometries (standard, medium, large)  in micrometers

  /// Field configuration as it was used in fits
  const float mFitElecEffFieldAbove = 2000.0;                                                  ///< Electric field above the GEM in V/cm (for extraction efficiency scans)
  const float mFitElecEffFieldBelow = 0.0;                                                     ///< Electric field below the GEM in V/cm (for collection efficiency scans)
  const float mFitElecEffPotentialGEM = 300.0;                                                 ///< Electric potential applied to GEM in volt
  const float mFitElecEffFieldGEM = mFitElecEffPotentialGEM / (mFitElecEffThickness * 0.0001); ///< Electric field inside of the GEM approximated as parallel plate capacitor in V/cm

  /// Scaling parameters from fits (standard, medium, large)
  const std::array<float, 3> mFitElecEffTuneEta1;      ///< Tuning of field ratio eta1 (also referred to as parameter s1)
  const std::array<float, 3> mFitElecEffTuneEta2;      ///< Tuning of field ratio eta2 (also referred to as parameter s2)
  const std::array<float, 3> mFitElecEffTuneDiffusion; ///< Tuning of geometric parameter C4 in order to implement diffusion (also referred to as parameter s3)

  /// Results from absolute gain simulations (standard, medium, large)
  const std::array<float, 3> mFitAbsGainConstant; ///< Constant from exponential fit function
  const std::array<float, 3> mFitAbsGainSlope;    ///< Slope from exponential fit function
  float mAbsGainScaling;                          ///< We allow a scaling factor of the gain curves for tuning (by default this factor is set to 1.0

  /// Results from single gain fluctuation simulations and fit to distribution (standard, medium, large)
  const std::array<float, 3> mFitSingleGainF0; ///< Value for f0 in single gain fluctuation distribution
  const std::array<float, 3> mFitSingleGainU0; ///< Value for U0 in single gain fluctuation distribution
  const std::array<float, 3> mFitSingleGainQ;  ///< Value for Q in single gain fluctuation distribution

  /// Some parameters are constant for a fixed GEM pitch, so we evaluate them once in the constructor (standard, medium, large)
  std::array<float, 3> mParamC1;
  std::array<float, 3> mParamC2;
  std::array<float, 3> mParamC3;
  std::array<float, 3> mParamC4;
  std::array<float, 3> mParamC5;
  std::array<float, 3> mParamC6;

  /// Properties of quadruple GEM stack
  std::array<int, 4> mGeometry;        ///< Array with GEM geometries (possible geometries are 0 standard, 1 medium, 2 large)
  std::array<float, 5> mDistance;      ///< Array with distances between cathode-GEM1, GEM1-GEM2, GEM2-GEM3, GEM3-GEM4, GEM4-anode (in cm)
  std::array<float, 4> mPotential;     ///< Array with GEM potentials (in volt)
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

  // Mean energy to create electron-ion pair in gas (here NeCO2N2 90-10-5, in eV)
  const float Wi = 37.3;

  // Fano factor for NeCO2N2 90-10-5 (Please check this!)
  const float Fano = 0.13;
};
} // namespace TPC
} // namespace o2

#endif // ALICEO2_TPC_ModelGEM_H_
