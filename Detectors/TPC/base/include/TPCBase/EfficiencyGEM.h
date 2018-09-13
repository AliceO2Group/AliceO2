// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file EfficiencyGEM.h
/// \brief Definition for the model calculations + simulations of the GEM efficiencies
/// \author Viktor Ratza, University of Bonn, ratza@hiskp.uni-bonn.de

// ================================================================================
// How are the efficiencies obtained?
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
// By selecting the GEM geometry [-> setGeometry(int geom)] all relevant
// geometry parameters and tuning parameters will be set which have been used for the fit.
// The efficiencies are thereupon calculated by the fitted equations.

// ================================================================================
// Remarks for naming of the variables:
// ================================================================================
//
// mNumberHoles (in PhD/paper: N)
//   Describes the number of holes for the 2D model calculations. The number
//   of holes is given by 2N-1, i.e. mNumberHoles=2 refers to 3 GEM holes (one
//   central GEM hole and two GEM holes at the outside).
//
// mGeometryThickness (in PhD/paper: d) [unit: micrometers]
//   Thickness of the GEM foil which is given by the thickness of the Polyimide
//   layer + 2x thickness of the Copper layers
//
// mGeometryPitch (in PhD/paper: p) [unit: micrometers]
//   Pitch of the GEM foil.
//
// mGeometryHoleDiameter (in PhD/paper: L) [unit: micrometers]
//   Hole diameter for the GEM hole. There is no differentiation between an inner
//   and an outer hole diameter for the 2D model calculations.
//
// mGeometryWidth (in PhD/paper: w) [unit: micrometers]
//   Describes the width for a unit cell (pitch) + 2x the distance to the end of the
//   GEM electrodes. This variable can be expressed in terms of the pitch and the hole
//   diameter according to w=2p-L. It is only used for internal calculations and no
//   definition is required by the user.
//
// mGeometryDistancePrevStage (in PhD/paper: g1) [unit: micrometers]
//   Here g1/2 describes the distance from the center of the GEM foil to the cathode
//   or the previous amplification stage.
//
// mGeometryDistanceNextStage (in PhD/paper: g2) [unit: micrometers]
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

#ifndef ALICEO2_TPC_EfficiencyGEM_H_
#define ALICEO2_TPC_EfficiencyGEM_H_

namespace o2
{
namespace TPC
{

/// \class EfficiencyGEM

class EfficiencyGEM
{
 public:
  /// Constructor
  EfficiencyGEM();

  /// Destructor
  ~EfficiencyGEM() = default;

  /// Set GEM geometry (by default we use a standard pitch GEM)
  /// \param geom Geometry of the GEM (1 standard, 2 medium, 3 large)
  void setGeometry(int geom);

  /// Get the electron collection efficiency of the GEM for the a given field ratio
  /// \param ElecFieldRatioAbove Ratio of the electric field above the GEM / the electric field inside of the GEM
  float getCollectionEfficiency(float ElecFieldRatioAbove);

  /// Get the electron extraction efficiency of the GEM for the a given field ratio
  /// \param ElecFieldRatioAbove Ratio of the electric field below the GEM / the electric field inside of the GEM
  float getExtractionEfficiency(float ElecFieldRatioBelow);

  /// Returns the thickness of the GEM (parameter d in GEM efficiency model) in micrometers
  const float getGeometryThickness();

 private:
  /// Geometric parameter C1 for collection efficiency
  float getParameterC1();

  /// Geometric parameter C2 for collection efficiency
  float getParameterC2();

  /// Geometric parameter C3 for collection efficiency
  float getParameterC3();

  /// Geometric parameter C4 for extraction efficiency
  float getParameterC4();

  /// Geometric parameter C5 for extraction efficiency
  float getParameterC5();

  /// Geometric parameter C6 for extraction efficiency
  float getParameterC6();

  /// Geometric parameter C7 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC7(float eta1, float eta2);

  /// Geometric parameter C8 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC8(float eta1, float eta2);

  /// Geometric parameter C9 for extraction efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC9(float eta1, float eta2);

  /// Geometric parameter C7Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC7Bar(float eta1, float eta2);

  /// Geometric parameter C8Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC8Bar(float eta1, float eta2);

  /// Geometric parameter C9Bar for collection efficiency as function of electric fields
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getParameterC9Bar(float eta1, float eta2);

  /// Geometric parameter C7Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getParameterC7BarFromX(float IntXStart, float IntXEnd);

  /// Geometric parameter C8Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// Integration limits same as for C7Bar
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getParameterC8BarFromX(float IntXStart, float IntXEnd);

  /// Geometric parameter C9Bar for collection efficiency as function of the integration limits on the top GEM electrode
  /// Integration limits same as for C7Bar
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getParameterC9BarFromX(float IntXStart, float IntXEnd);

  /// Flux at cathode: Term in front of lambda [sum of getLambdaCathodeF2 and getLambdaCathodef2]
  float getLambdaCathode();

  /// Flux at cathode: Term in front of lambda: Basic term for central GEM hole
  float getLambdaCathodef2();

  /// Flux at cathode: Term in front of lambda: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  float getLambdaCathodeF2(int n);

  /// Flux at cathode: Term in front of mu1 [sum of getMu1CathodeF2 and getMu1Cathodef2]
  float getMu1Cathode();

  /// Flux at cathode: Term in front of mu1: Basic term for central GEM hole
  float getMu1Cathodef2();

  /// Flux at cathode: Term in front of mu1: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  float getMu1CathodeF2(int n);

  /// Flux at cathode: Term in front of mu2 [sum of getMu2CathodeF2 and getMu2Cathodef2]
  float getMu2Cathode();

  /// Flux at cathode: Term in front of mu2: Basic term for central GEM hole
  float getMu2Cathodef2();

  /// Flux at cathode: Term in front of mu2: Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  float getMu2CathodeF2(int n);

  /// Flux C at top electrode: Term in front of (mu2-lambda) [sum of getMu2TopF2 and getMu2Topf2]
  /// For region 1 (before the kink) we integrate from -(w+L)/4 to -(w+L)/4 (no distance)
  /// For region 2 (within the kink) we integrate from -(w+L)/4 to return value of getIntXEndTop(float eta1, float eta2)
  /// For region 3 (after the kink) we integrate from -(w+L)/4 to -L/2 (hole top electrode of unit cell)
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getMu2Top(float IntXStart, float IntXEnd);

  /// Flux C at top electrode: Term in front of (mu2-lambda): Basic term for central GEM hole
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getMu2Topf2(float IntXStart, float IntXEnd);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Basic term for central GEM hole: Taylor expansion at -(w+L)/4: Order 0
  float getMu2TopfTaylorTerm0();

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Basic term for central GEM hole: Taylor expansion at -(w+L)/4: Order 2
  float getMu2TopfTaylorTerm2();

  /// Flux C at top electrode: Term in front of (mu2-lambda): Additional terms for outer GEM holes (2N-1 holes in total)
  /// \param n Summation index where n=2..N
  /// \param IntXStart Start value for x integration
  /// \param IntXEnd End value for x integration
  float getMu2TopF2(int n, float IntXStart, float IntXEnd);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Additional terms for outer GEM holes (2N-1 holes in total): Taylor expansion at -(w+L)/4: Order 0
  float getMu2TopFTaylorTerm0(int n);

  /// Electric field (y component) at top electrode: Term in front of (mu2-lambda):
  /// Additional terms for outer GEM holes (2N-1 holes in total): Taylor expansion at -(w+L)/4: Order 2
  float getMu2TopFTaylorTerm2(int n);

  /// Returns the x position on the bottom electrode of the GEM where the sign flip of the electric field in y direction
  /// appears
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getIntXEndBot(float eta1, float eta2);

  /// Returns the x position on the top electrode of the GEM where the sign flip of the electric field in y direction
  /// appears
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getIntXEndTop(float eta1, float eta2);

  /// Field ratio Eta1 for the collection efficiency where the kink starts (i.e. the plateau region ends)
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getEta1Kink1(float eta2);

  /// Field ratio Eta1 for the collection efficiency where the kink ends
  /// \param eta2 Ratio of electric fields: Below GEM / Field in GEM hole
  float getEta1Kink2(float eta2);

  /// Field ratio Eta2 for the extraction efficiency where the kink starts
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  float getEta2Kink1(float eta1);

  /// Field ratio Eta2 for the extraction efficiency where the kink ends
  /// \param eta1 Ratio of electric fields: Above GEM / Field in GEM hole
  float getEta2Kink2(float eta1);

  /// Electric field (y component) at top electrode: Q term in front of (mu2-lambda):
  /// Taylor expansion: Order 0 [sum of getMu2TopfTaylorTerm0 and getMu2TopFTaylorTerm0]
  float getHtop0();

  /// Electric field (y component) at top electrode: Q term in front of (mu2-lambda):
  /// Taylor expansion: Order 2 [sum of getMu2TopfTaylorTerm2 and getMu2TopFTaylorTerm2]
  float getHtop2();

  /// Due to geometric symmetries the parameters C7, C8 and C9 can be calculated just like C7Bar, C8Bar
  /// and C9Bar if we "flip" the cathode and the anode. We do this by flipping the distances
  /// mGeometryDistancePrevStage and mGeometryDistanceNextStage and use the same equations as for
  /// C7Bar, C8Bar and C9Bar.
  void flipDistanceNextPrevStage();

  /// GEM geometry definitions as it was used in fits
  const int mNumberHoles = 1000;         ///< Number of GEM holes
  const float mGeometryThickness = 60.0; ///< Thickness of the GEM in micrometers
  const float mGeometryHoleDiameter =
    70.0; ///< Diameter of the GEM hole (there is no differentiation between inner and outer diameter) in micrometers
  float mGeometryDistancePrevStage =
    400.0; ///< 2*Distance from center of GEM to previous stage (i.e. cathode or GEM) in micrometers
  float mGeometryDistanceNextStage =
    400.0;              ///< 2*Distance from center of GEM to next stage (i.e. anode or GEM) in micrometers
  float mGeometryPitch; ///< Pitch of the GEM in micrometers
  float mGeometryWidth; ///< 2*Pitch - Hole diameter in micrometers

  /// Field configuration as it was used in fits
  const float mElecFieldAbove = 2000.0; ///< Electric field above the GEM in Volts/cm
  const float mElecFieldBelow = 0.0;    ///< Electric field below the GEM in Volts/cm
  const float mPotentialGEM = 300.0;    ///< Electric potential applied to GEM in Volts
  const float mElecFieldGEM =
    mPotentialGEM /
    (mGeometryThickness * 0.0001); ///< Electric field inside of GEM approximated as parallel plate capacitor

  /// Scaling parameters from fits
  float mGeometryTuneEta1;      ///< Tuning of field ratio eta1
  float mGeometryTuneEta2;      ///< Tuning of field ratio eta2
  float mGeometryTuneDiffusion; ///< Tuning of geometric parameter C4 in order to implement diffusion

  /// Some parameters are constant for a fixed GEM pitch, so we evaluate them once after
  /// the pitch has been set or altered
  float mParamC1;
  float mParamC2;
  float mParamC3;
  float mParamC4;
  float mParamC5;
  float mParamC6;

  static constexpr float sPi = 3.1415926;

  // Deviation factor kappa which describes the divergence of the field of a parallel plate
  // capacitor to the average electric field inside of a GEM hole
  static constexpr float sKappa = 0.514;
};
}
}

#endif // ALICEO2_TPC_EfficiencyGEM_H_
