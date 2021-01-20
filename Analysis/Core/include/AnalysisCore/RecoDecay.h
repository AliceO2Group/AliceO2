// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file RecoDecay.h
/// \brief Implementation of the RecoDecay class.
///
/// \author Vít Kučera <vit.kucera@cern.ch>, CERN

#ifndef O2_ANALYSIS_RECODECAY_H_
#define O2_ANALYSIS_RECODECAY_H_

#include <tuple>
#include <vector>
#include <array>
#include <cmath>

#include <TDatabasePDG.h>
#include <TPDGCode.h>

#include "CommonConstants/MathConstants.h"
#include "Framework/Logger.h"

using std::array;
using namespace o2::constants::math;

/// Base class for calculating properties of reconstructed decays.

class RecoDecay
{
 public:
  /// Default constructor
  RecoDecay() = default;

  /// Default destructor
  ~RecoDecay() = default;

  // Auxiliary functions

  /// Sums numbers.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return sum of numbers
  template <typename... T>
  static auto sum(const T&... args)
  {
    return (args + ...);
  }

  /// Squares a number.
  /// \note Promotes number to double before squaring to avoid precision loss in float multiplication.
  /// \param num  a number of arbitrary type
  /// \return number squared
  template <typename T>
  static double sq(T num)
  {
    return (double)num * (double)num;
  }

  /// Sums squares of numbers.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return sum of squares of numbers
  template <typename... T>
  static double sumOfSquares(const T&... args)
  {
    return (((double)args * (double)args) + ...);
  }

  /// Calculates square root of a sum of squares of numbers.
  /// \param args  arbitrary number of numbers of arbitrary types
  /// \return square root of sum of squares of numbers
  template <typename... T>
  static double sqrtSumOfSquares(const T&... args)
  {
    return std::sqrt(sumOfSquares(args...));
  }

  /// Sums i-th elements of containers.
  /// \param index  element index
  /// \param args  pack of containers of elements accessible by index
  /// \return sum of i-th elements
  template <typename... T>
  static auto getElement(int index, const T&... args)
  {
    return (args[index] + ...);
  }

  /// Sums 3-vectors.
  /// \param args  pack of 3-vector arrays
  /// \return sum of vectors
  template <typename... T>
  static auto sumOfVec(const array<T, 3>&... args)
  {
    return array{getElement(0, args...), getElement(1, args...), getElement(2, args...)};
  }

  /// Calculates scalar product of vectors.
  /// \note Promotes numbers to double to avoid precision loss in float multiplication.
  /// \param N  dimension
  /// \param vec1,vec2  vectors
  /// \return scalar product
  template <std::size_t N, typename T, typename U>
  static double dotProd(const array<T, N>& vec1, const array<U, N>& vec2)
  {
    double res{0};
    for (auto iDim = 0; iDim < N; ++iDim) {
      res += (double)vec1[iDim] * (double)vec2[iDim];
    }
    return res;
  }

  /// FIXME: probably cross and dot products should be in some utility class
  /// Calculates cross product of vectors in three dimensions.
  /// \note Promotes numbers to double to avoid precision loss in float multiplication.
  /// \param vec1,vec2  vectors
  /// \return cross-product vector
  template <typename T, typename U>
  static array<double, 3> crossProd(const array<T, 3>& vec1, const array<U, 3>& vec2)
  {
    return array<double, 3>{((double)vec1[1] * (double)vec2[2]) - ((double)vec1[2] * (double)vec2[1]),
                            ((double)vec1[2] * (double)vec2[0]) - ((double)vec1[0] * (double)vec2[2]),
                            ((double)vec1[0] * (double)vec2[1]) - ((double)vec1[1] * (double)vec2[0])};
  }

  /// Calculates magnitude squared of a vector.
  /// \param N  dimension
  /// \param vec  vector
  /// \return magnitude squared
  template <std::size_t N, typename T>
  static double mag2(const array<T, N>& vec)
  {
    return dotProd(vec, vec);
  }

  /// Calculates 3D distance between two points.
  /// \param point1,point2  {x, y, z} coordinates of points
  /// \return 3D distance between two points
  template <typename T, typename U>
  static double distance(const T& point1, const U& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1], point1[2] - point2[2]);
  }

  /// Calculates 2D {x, y} distance between two points.
  /// \param point1,point2  {x, y, z} or {x, y} coordinates of points
  /// \return 2D {x, y} distance between two points
  template <typename T, typename U>
  static double distanceXY(const T& point1, const U& point2)
  {
    return sqrtSumOfSquares(point1[0] - point2[0], point1[1] - point2[1]);
  }

  // Calculation of kinematic quantities

  /// Calculates pseudorapidity.
  /// \param mom  3-momentum array
  /// \return pseudorapidity
  template <typename T>
  static double Eta(const array<T, 3>& mom)
  {
    // eta = arctanh(pz/p)
    if (std::abs(mom[0]) < Almost0 && std::abs(mom[1]) < Almost0) { // very small px and py
      return (double)(mom[2] > 0 ? VeryBig : -VeryBig);
    }
    return (double)(std::atanh(mom[2] / P(mom)));
  }

  /// Calculates rapidity.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \return rapidity
  template <typename T, typename U>
  static double Y(const array<T, 3>& mom, U mass)
  {
    // y = arctanh(pz/E)
    return std::atanh(mom[2] / E(mom, mass));
  }

  /// Calculates azimuthal angle from x and y momentum components.
  /// \param px,py  {x, y} momentum components
  /// \return azimuthal angle
  static double Phi(double px, double py)
  {
    // phi = pi+TMath::Atan2(-py,-px)
    return (double)(o2::constants::math::PI + std::atan2(-py, -px));
  }

  /// Calculates azimuthal angle from 3-(or 2-)momenta.
  /// \param args  pack of 3-(or 2-)momentum arrays
  /// \return azimuthal angle
  template <std::size_t N, typename T>
  static double Phi(const array<T, N>& vec)
  {
    return Phi(vec[0], vec[1]);
  }

  /// Calculates cosine of pointing angle.
  /// \param posPV  {x, y, z} position of the primary vertex
  /// \param posSV  {x, y, z} position of the secondary vertex
  /// \param mom  3-momentum array
  /// \return cosine of pointing angle
  template <typename T, typename U, typename V>
  static double CPA(const T& posPV, const U& posSV, const array<V, 3>& mom)
  {
    // CPA = (l . p)/(|l| |p|)
    auto lineDecay = array{posSV[0] - posPV[0], posSV[1] - posPV[1], posSV[2] - posPV[2]};
    auto cos = dotProd(lineDecay, mom) / std::sqrt(mag2(lineDecay) * mag2(mom));
    if (cos < -1.) {
      return -1.;
    }
    if (cos > 1.) {
      return 1.;
    }
    return cos;
  }

  /// Calculates cosine of pointing angle in the {x, y} plane.
  /// \param posPV  {x, y, z} or {x, y} position of the primary vertex
  /// \param posSV  {x, y, z} or {x, y} position of the secondary vertex
  /// \param mom  {x, y, z} or {x, y} momentum array
  /// \return cosine of pointing angle in {x, y}
  template <std::size_t N, typename T, typename U, typename V>
  static double CPAXY(const T& posPV, const U& posSV, const array<V, N>& mom)
  {
    // CPAXY = (r . pT)/(|r| |pT|)
    auto lineDecay = array{posSV[0] - posPV[0], posSV[1] - posPV[1]};
    auto momXY = array{mom[0], mom[1]};
    auto cos = dotProd(lineDecay, momXY) / std::sqrt(mag2(lineDecay) * mag2(momXY));
    if (cos < -1.) {
      return -1.;
    }
    if (cos > 1.) {
      return 1.;
    }
    return cos;
  }

  /// Calculates proper lifetime times c.
  /// \note Promotes numbers to double before squaring to avoid precision loss in float multiplication.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \param length  decay length
  /// \return proper lifetime times c
  template <typename T, typename U, typename V>
  static double Ct(const array<T, 3>& mom, U length, V mass)
  {
    // c t = l m c^2/(p c)
    return (double)length * (double)mass / P(mom);
  }

  /// Calculates cosine of θ* (theta star).
  /// \note Implemented for 2 prongs only.
  /// \param arrMom  array of two 3-momentum arrays
  /// \param arrMass  array of two masses (in the same order as arrMom)
  /// \param mTot  assumed mass of mother particle
  /// \param iProng  index of the prong
  /// \return cosine of θ* of the i-th prong under the assumption of the invariant mass
  template <typename T, typename U, typename V>
  static double CosThetaStar(const array<array<T, 3>, 2>& arrMom, const array<U, 2>& arrMass, V mTot, int iProng)
  {
    auto pVecTot = PVec(arrMom[0], arrMom[1]);                                                                             // momentum of the mother particle
    auto pTot = P(pVecTot);                                                                                                // magnitude of the momentum of the mother particle
    auto eTot = E(pTot, mTot);                                                                                             // energy of the mother particle
    auto gamma = eTot / mTot;                                                                                              // γ, Lorentz gamma factor of the mother particle
    auto beta = pTot / eTot;                                                                                               // β, velocity of the mother particle
    auto pStar = std::sqrt(sq(sq(mTot) - sq(arrMass[0]) - sq(arrMass[1])) - sq(2 * arrMass[0] * arrMass[1])) / (2 * mTot); // p*, prong momentum in the rest frame of the mother particle
    // p* = √[(M^2 - m1^2 - m2^2)^2 - 4 m1^2 m2^2]/2M
    // Lorentz transformation of the longitudinal momentum of the prong into the detector frame:
    // p_L,i = γ (p*_L,i + β E*_i)
    // p*_L,i = p_L,i/γ - β E*_i
    // cos(θ*_i) = (p_L,i/γ - β E*_i)/p*
    return (dotProd(arrMom[iProng], pVecTot) / (pTot * gamma) - beta * E(pStar, arrMass[iProng])) / pStar;
  }

  /// Sums 3-momenta.
  /// \param args  pack of 3-momentum arrays
  /// \return total 3-momentum array
  template <typename... T>
  static auto PVec(const array<T, 3>&... args)
  {
    return sumOfVec(args...);
  }

  /// Calculates momentum squared from momentum components.
  /// \param px,py,pz  {x, y, z} momentum components
  /// \return momentum squared
  static double P2(double px, double py, double pz)
  {
    return sumOfSquares(px, py, pz);
  }

  /// Calculates total momentum squared of a sum of 3-momenta.
  /// \param args  pack of 3-momentum arrays
  /// \return total momentum squared
  template <typename... T>
  static double P2(const array<T, 3>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...), getElement(2, args...));
  }

  /// Calculates (total) momentum magnitude.
  /// \param args  {x, y, z} momentum components or pack of 3-momentum arrays
  /// \return (total) momentum magnitude
  template <typename... T>
  static double P(const T&... args)
  {
    return std::sqrt(P2(args...));
  }

  /// Calculates transverse momentum squared from momentum components.
  /// \param px,py  {x, y} momentum components
  /// \return transverse momentum squared
  static double Pt2(double px, double py)
  {
    return sumOfSquares(px, py);
  }

  /// Calculates total transverse momentum squared of a sum of 3-(or 2-)momenta.
  /// \param args  pack of 3-(or 2-)momentum arrays
  /// \return total transverse momentum squared
  template <std::size_t N, typename... T>
  static double Pt2(const array<T, N>&... args)
  {
    return sumOfSquares(getElement(0, args...), getElement(1, args...));
  }

  /// Calculates (total) transverse momentum.
  /// \param args  {x, y} momentum components or pack of 3-momentum arrays
  /// \return (total) transverse momentum
  template <typename... T>
  static double Pt(const T&... args)
  {
    return std::sqrt(Pt2(args...));
  }

  /// Calculates energy squared from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \return energy squared
  template <typename... T>
  static double E2(T... args)
  {
    return sumOfSquares(args...);
  }

  /// Calculates energy squared from momentum vector and mass.
  /// \param mom  3-momentum array
  /// \param mass  mass
  /// \return energy squared
  template <typename T, typename U>
  static double E2(const array<T, 3>& mom, U mass)
  {
    return E2(mom[0], mom[1], mom[2], mass);
  }

  /// Calculates energy from momentum and mass.
  /// \param args  momentum magnitude, mass
  /// \param args  {x, y, z} momentum components, mass
  /// \param args  3-momentum array, mass
  /// \return energy
  template <typename... T>
  static double E(const T&... args)
  {
    return std::sqrt(E2(args...));
  }

  /// Calculates invariant mass squared from momentum magnitude and energy.
  /// \param mom  momentum magnitude
  /// \param energy  energy
  /// \return invariant mass squared
  static double M2(double mom, double energy)
  {
    return energy * energy - mom * mom;
  }

  /// Calculates invariant mass squared from momentum aray and energy.
  /// \param mom  3-momentum array
  /// \param energy  energy
  /// \return invariant mass squared
  template <typename T>
  static double M2(const array<T, 3>& mom, double energy)
  {
    return energy * energy - P2(mom);
  }

  /// Calculates invariant mass squared from momenta and masses of several particles (prongs).
  /// \param N  number of prongs
  /// \param arrMom  array of N 3-momentum arrays
  /// \param arrMass  array of N masses (in the same order as arrMom)
  /// \return invariant mass squared
  template <std::size_t N, typename T, typename U>
  static double M2(const array<array<T, 3>, N>& arrMom, const array<U, N>& arrMass)
  {
    array<double, 3> momTotal{0., 0., 0.}; // candidate momentum vector
    double energyTot{0.};                  // candidate energy
    for (auto iProng = 0; iProng < N; ++iProng) {
      for (auto iMom = 0; iMom < 3; ++iMom) {
        momTotal[iMom] += arrMom[iProng][iMom];
      } // loop over momentum components
      energyTot += E(arrMom[iProng], arrMass[iProng]);
    } // loop over prongs
    return energyTot * energyTot - P2(momTotal);
  }

  /// Calculates invariant mass.
  /// \param args  momentum magnitude, energy
  /// \param args  3-momentum array, energy
  /// \param args  array of momenta, array of masses
  /// \return invariant mass
  template <typename... T>
  static double M(const T&... args)
  {
    return std::sqrt(M2(args...));
  }

  // Calculation of topological quantities

  /// Calculates impact parameter in the bending plane of the particle w.r.t. a point
  /// \param point  {x, y, z} position of the point
  /// \param posSV  {x, y, z} position of the secondary vertex
  /// \param mom  {x, y, z} particle momentum array
  /// \return impact parameter in {x, y}
  template <typename T, typename U, typename V>
  static double ImpParXY(const T& point, const U& posSV, const array<V, 3>& mom)
  {
    // Ported from AliAODRecoDecay::ImpParXY
    auto flightLineXY = array{posSV[0] - point[0], posSV[1] - point[1]};
    auto k = dotProd(flightLineXY, array{mom[0], mom[1]}) / Pt2(mom);
    auto dx = flightLineXY[0] - k * (double)mom[0];
    auto dy = flightLineXY[1] - k * (double)mom[1];
    auto absImpPar = sqrtSumOfSquares(dx, dy);
    auto flightLine = array{posSV[0] - point[0], posSV[1] - point[1], posSV[2] - point[2]};
    auto cross = crossProd(mom, flightLine);
    return (cross[2] > 0. ? absImpPar : -1. * absImpPar);
  }

  /// Calculates the difference between measured and expected track impact parameter
  /// normalized to its uncertainty
  /// \param decLenXY decay lenght in {x, y} plane
  /// \param errDecLenXY error on decay lenght in {x, y} plane
  /// \param momMother {x, y, z} or {x, y} candidate momentum array
  /// \param impParProng prong impact parameter
  /// \param errImpParProng error on prong impact parameter
  /// \param momProng {x, y, z} or {x, y} prong momentum array
  /// \return normalized difference between expected and observed impact parameter
  template <std::size_t N, std::size_t M, typename T, typename U, typename V, typename W, typename X, typename Y>
  static double normImpParMeasMinusExpProng(T decLenXY, U errDecLenXY, const array<V, N>& momMother, W impParProng,
                                            X errImpParProng, const array<Y, M>& momProng)
  {
    // Ported from AliAODRecoDecayHF::Getd0MeasMinusExpProng adding normalization directly in the function
    auto sinThetaP = ((double)momProng[0] * (double)momMother[1] - (double)momProng[1] * (double)momMother[0]) /
                     (Pt(momProng) * Pt(momMother));
    auto diff = impParProng - (double)decLenXY * sinThetaP;
    auto errImpParExpProng = (double)errDecLenXY * sinThetaP;
    auto errDiff = sqrtSumOfSquares(errImpParProng, errImpParExpProng);
    return (errDiff > 0. ? diff / errDiff : 0.);
  }

  /// Calculates maximum normalized difference between measured and expected impact parameter of candidate prongs
  /// \param posPV {x, y, z} or {x, y} position of primary vertex
  /// \param posSV {x, y, z} or {x, y} position of secondary vertex
  /// \param errDecLenXY error on decay lenght in {x, y} plane
  /// \param momMother {x, y, z} or {x, y} candidate momentum array
  /// \param arrImpPar array of prong impact parameters
  /// \param arrErrImpPar array of errors on prong impact parameter (same order as arrImpPar)
  /// \param momMom array of {x, y, z} or {x, y} prong momenta (same order as arrImpPar)
  /// \return maximum normalized difference between expected and observed impact parameters
  template <std::size_t N, std::size_t M, std::size_t K, typename T, typename U, typename V, typename W, typename X,
            typename Y, typename Z>
  static double maxNormalisedDeltaIP(const T& posPV, const U& posSV, V errDecLenXY, const array<W, M>& momMother,
                                     const array<X, N>& arrImpPar, const array<Y, N>& arrErrImpPar,
                                     const array<array<Z, K>, N>& arrMom)
  {
    auto decLenXY = distanceXY(posPV, posSV);
    double maxNormDeltaIP{0.};
    for (auto iProng = 0; iProng < N; ++iProng) {
      auto prongNormDeltaIP = normImpParMeasMinusExpProng(decLenXY, errDecLenXY, momMother, arrImpPar[iProng],
                                                          arrErrImpPar[iProng], arrMom[iProng]);
      if (std::abs(prongNormDeltaIP) > std::abs(maxNormDeltaIP)) {
        maxNormDeltaIP = prongNormDeltaIP;
      }
    }
    return maxNormDeltaIP;
  }

  /// Returns particle mass based on PDG code.
  /// \param pdg  PDG code
  /// \return particle mass
  static double getMassPDG(int pdg)
  {
    // Try to get the particle mass from the list first.
    for (const auto& particle : mListMass) {
      if (std::get<0>(particle) == pdg) {
        return std::get<1>(particle);
      }
    }
    // Get the mass of the new particle and add it in the list.
    auto newMass = TDatabasePDG::Instance()->GetParticle(pdg)->Mass();
    mListMass.push_back(std::make_tuple(pdg, newMass));
    return newMass;
  }

  /// Finds the mother of an MC particle by looking for the expected PDG code in the mother chain.
  /// \param particlesMC  table with MC particles
  /// \param particle  MC particle
  /// \param PDGMother  expected mother PDG code
  /// \param acceptAntiParticles  switch to accept the antiparticle of the expected mother
  /// \param sign  antiparticle indicator of the found mother w.r.t. PDGMother; 1 if particle, -1 if antiparticle, 0 if mother not found
  /// \return index of the mother particle if found, -1 otherwise
  template <typename T>
  static int getMother(const T& particlesMC, const typename T::iterator& particle, int PDGMother, bool acceptAntiParticles = false, int8_t* sign = nullptr)
  {
    int8_t sgn = 0;                 // 1 if the expected mother is particle, -1 if antiparticle (w.r.t. PDGMother)
    int indexMother = -1;           // index of the final matched mother, if found
    auto particleMother = particle; // Initialise loop over mothers.
    int stage = 0;                  // mother tree level (just for debugging)
    if (sign) {
      *sign = sgn;
    }
    while (particleMother.mother0() > -1) {
      auto indexMotherTmp = particleMother.mother0();
      particleMother = particlesMC.iteratorAt(indexMotherTmp);
      // Check mother's PDG code.
      auto PDGParticleIMother = particleMother.pdgCode(); // PDG code of the mother
      //printf("getMother: ");
      //for (int i = stage; i < 0; i++) // Indent to make the tree look nice.
      //  printf(" ");
      //printf("Stage %d: Mother PDG: %d\n", stage, PDGParticleIMother);
      if (PDGParticleIMother == PDGMother) { // exact PDG match
        sgn = 1;
        indexMother = indexMotherTmp;
        break;
      } else if (acceptAntiParticles && PDGParticleIMother == -PDGMother) { // antiparticle PDG match
        sgn = -1;
        indexMother = indexMotherTmp;
        break;
      }
      stage--;
    }
    if (sign) {
      *sign = sgn;
    }
    return indexMother;
  }

  /// Gets the complete list of indices of final-state daughters of an MC particle.
  /// \param particlesMC  table with MC particles
  /// \param index  index of the MC particle
  /// \param list  vector where the indices of final-state daughters will be added
  /// \param arrPDGFinal  array of PDG codes of particles to be considered final
  /// \param depthMax  maximum decay tree level; Daughters at this level (or beyond) will be considered final. If -1, all levels are considered.
  /// \param stage  decay tree level; If different from 0, the particle itself will be added in the list in case it has no daughters.
  /// \note Final state is defined as particles from arrPDGFinal plus final daughters of any other decay branch.
  template <std::size_t N, typename T>
  static void getDaughters(const T& particlesMC,
                           int index,
                           std::vector<int>* list,
                           const array<int, N>& arrPDGFinal,
                           int depthMax = -1,
                           int8_t stage = 0)
  {
    if (index <= -1) {
      //Printf("getDaughters: Error: No particle: index %d", index);
      return;
    }
    if (!list) {
      //Printf("getDaughters: Error: No list!");
      return;
    }
    // Get the particle.
    auto particle = particlesMC.iteratorAt(index);
    bool isFinal = false;                     // Flag to indicate the end of recursion
    if (depthMax > -1 && stage >= depthMax) { // Maximum depth has been reached (or exceeded).
      isFinal = true;
    }
    // Get the range of daughter indices.
    int indexDaughterFirst = particle.daughter0();
    int indexDaughterLast = particle.daughter1();
    // Check whether there are any daughters.
    if (!isFinal && indexDaughterFirst <= -1 && indexDaughterLast <= -1) {
      // If the original particle has no daughters, we do nothing and exit.
      if (stage == 0) {
        //Printf("getDaughters: No daughters of %d", index);
        return;
      }
      // If this is not the original particle, we are at the end of this branch and this particle is final.
      isFinal = true;
    }
    // If the particle has daughters but is considered to be final, we label it as final.
    auto PDGParticle = particle.pdgCode();
    if (!isFinal) {
      for (auto PDGi : arrPDGFinal) {
        if (std::abs(PDGParticle) == std::abs(PDGi)) { // Accept antiparticles.
          isFinal = true;
          break;
        }
      }
    }
    // If the particle is labelled as final, we add this particle in the list of final daughters and exit.
    if (isFinal) {
      //printf("getDaughters: ");
      //for (int i = 0; i < stage; i++) // Indent to make the tree look nice.
      //  printf(" ");
      //printf("Stage %d: Adding %d (PDG %d) as final daughter.\n", stage, index, PDGParticle);
      list->push_back(index);
      return;
    }
    // If we are here, we have to follow the daughter tree.
    //printf("getDaughters: ");
    //for (int i = 0; i < stage; i++) // Indent to make the tree look nice.
    //  printf(" ");
    //printf("Stage %d: %d (PDG %d) -> %d-%d\n", stage, index, PDGParticle, indexDaughterFirst, indexDaughterLast);
    // Call itself to get daughters of daughters recursively.
    // Get daughters of the first daughter.
    if (indexDaughterFirst > -1) {
      getDaughters(particlesMC, indexDaughterFirst, list, arrPDGFinal, depthMax, stage + 1);
    }
    // Get daughters of the daughters in between if any.
    // Daughter indices are supposed to be consecutive and in increasing order.
    // Reverse order means two daughters.
    if (indexDaughterFirst > -1 && indexDaughterLast > -1) {
      for (auto iD = indexDaughterFirst + 1; iD < indexDaughterLast; ++iD) {
        getDaughters(particlesMC, iD, list, arrPDGFinal, depthMax, stage + 1);
      }
    }
    // Get daughters of the last daughter if different from the first one.
    // Same indices indicate a single daughter.
    if (indexDaughterLast > -1 && indexDaughterLast != indexDaughterFirst) {
      getDaughters(particlesMC, indexDaughterLast, list, arrPDGFinal, depthMax, stage + 1);
    }
  }

  /// Checks whether the reconstructed decay candidate is the expected decay.
  /// \param particlesMC  table with MC particles
  /// \param arrDaughters  array of candidate daughters
  /// \param PDGMother  expected mother PDG code
  /// \param arrPDGDaughters  array of expected daughter PDG codes
  /// \param acceptAntiParticles  switch to accept the antiparticle version of the expected decay
  /// \param sign  antiparticle indicator of the found mother w.r.t. PDGMother; 1 if particle, -1 if antiparticle, 0 if mother not found
  /// \param depthMax  maximum decay tree level to check; Daughters up to this level will be considered. If -1, all levels are considered.
  /// \return index of the mother particle if the mother and daughters are correct, -1 otherwise
  template <std::size_t N, typename T, typename U>
  static int getMatchedMCRec(const T& particlesMC, const array<U, N>& arrDaughters, int PDGMother, array<int, N> arrPDGDaughters, bool acceptAntiParticles = false, int8_t* sign = nullptr, int depthMax = 1)
  {
    //Printf("MC Rec: Expected mother PDG: %d", PDGMother);
    int8_t sgn = 0;                        // 1 if the expected mother is particle, -1 if antiparticle (w.r.t. PDGMother)
    int indexMother = -1;                  // index of the mother particle
    std::vector<int> arrAllDaughtersIndex; // vector of indices of all daughters of the mother of the first provided daughter
    array<int, N> arrDaughtersIndex;       // array of indices of provided daughters
    if (sign) {
      *sign = sgn;
    }
    // Loop over decay candidate prongs
    for (auto iProng = 0; iProng < N; ++iProng) {
      auto particleI = arrDaughters[iProng].label(); // ith daughter particle
      arrDaughtersIndex[iProng] = particleI.globalIndex();
      // Get the list of daughter indices from the mother of the first prong.
      if (iProng == 0) {
        // Get the mother index and its sign.
        // PDG code of the first daughter's mother determines whether the expected mother is a particle or antiparticle.
        indexMother = getMother(particlesMC, particleI, PDGMother, acceptAntiParticles, &sgn);
        // Check whether mother was found.
        if (indexMother <= -1) {
          //Printf("MC Rec: Rejected: bad mother index or PDG");
          return -1;
        }
        //Printf("MC Rec: Good mother: %d", indexMother);
        auto particleMother = particlesMC.iteratorAt(indexMother);
        int indexDaughterFirst = particleMother.daughter0(); // index of the first direct daughter
        int indexDaughterLast = particleMother.daughter1();  // index of the last direct daughter
        // Check the daughter indices.
        if (indexDaughterFirst <= -1 && indexDaughterLast <= -1) {
          //Printf("MC Rec: Rejected: bad daughter index range: %d-%d", indexDaughterFirst, indexDaughterLast);
          return -1;
        }
        // Check that the number of direct daughters is not larger than the number of expected final daughters.
        if (indexDaughterFirst > -1 && indexDaughterLast > -1 && indexDaughterLast - indexDaughterFirst + 1 > N) {
          //Printf("MC Rec: Rejected: too many direct daughters: %d (expected %d final)", indexDaughterLast - indexDaughterFirst + 1, N);
          return -1;
        }
        // Get the list of actual final daughters.
        getDaughters(particlesMC, indexMother, &arrAllDaughtersIndex, arrPDGDaughters, depthMax);
        //printf("MC Rec: Mother %d has %d final daughters:", indexMother, arrAllDaughtersIndex.size());
        //for (auto i : arrAllDaughtersIndex) {
        //  printf(" %d", i);
        //}
        //printf("\n");
        // Check whether the number of actual final daughters is equal to the number of expected final daughters (i.e. the number of provided prongs).
        if (arrAllDaughtersIndex.size() != N) {
          //Printf("MC Rec: Rejected: incorrect number of final daughters: %d (expected %d)", arrAllDaughtersIndex.size(), N);
          return -1;
        }
      }
      // Check that the daughter is in the list of final daughters.
      // (Check that the daughter is not a stepdaughter, i.e. particle pointing to the mother while not being its daughter.)
      bool isDaughterFound = false; // Is the index of this prong among the remaining expected indices of daughters?
      for (auto iD = 0; iD < arrAllDaughtersIndex.size(); ++iD) {
        if (arrDaughtersIndex[iProng] == arrAllDaughtersIndex[iD]) {
          arrAllDaughtersIndex[iD] = -1; // Remove this index from the array of expected daughters. (Rejects twin daughters, i.e. particle considered twice as a daughter.)
          isDaughterFound = true;
          break;
        }
      }
      if (!isDaughterFound) {
        //Printf("MC Rec: Rejected: bad daughter index: %d not in the list of final daughters", arrDaughtersIndex[iProng]);
        return -1;
      }
      // Check daughter's PDG code.
      auto PDGParticleI = particleI.pdgCode(); // PDG code of the ith daughter
      //Printf("MC Rec: Daughter %d PDG: %d", iProng, PDGParticleI);
      bool isPDGFound = false; // Is the PDG code of this daughter among the remaining expected PDG codes?
      for (auto iProngCp = 0; iProngCp < N; ++iProngCp) {
        if (PDGParticleI == sgn * arrPDGDaughters[iProngCp]) {
          arrPDGDaughters[iProngCp] = 0; // Remove this PDG code from the array of expected ones.
          isPDGFound = true;
          break;
        }
      }
      if (!isPDGFound) {
        //Printf("MC Rec: Rejected: bad daughter PDG: %d", PDGParticleI);
        return -1;
      }
    }
    //Printf("MC Rec: Accepted: m: %d", indexMother);
    if (sign) {
      *sign = sgn;
    }
    return indexMother;
  }

  /// Checks whether the MC particle is the expected one.
  /// \param particlesMC  table with MC particles
  /// \param candidate  candidate MC particle
  /// \param PDGParticle  expected particle PDG code
  /// \param acceptAntiParticles  switch to accept the antiparticle
  /// \return true if PDG code of the particle is correct, false otherwise
  template <typename T, typename U>
  static int isMatchedMCGen(const T& particlesMC, const U& candidate, int PDGParticle, bool acceptAntiParticles = false)
  {
    array<int, 0> arrPDGDaughters;
    return isMatchedMCGen(particlesMC, candidate, PDGParticle, std::move(arrPDGDaughters), acceptAntiParticles);
  }

  /// Check whether the MC particle is the expected one and whether it decayed via the expected decay channel.
  /// \param particlesMC  table with MC particles
  /// \param candidate  candidate MC particle
  /// \param PDGParticle  expected particle PDG code
  /// \param arrPDGDaughters  array of expected PDG codes of daughters
  /// \param acceptAntiParticles  switch to accept the antiparticle
  /// \param sign  antiparticle indicator of the candidate w.r.t. PDGParticle; 1 if particle, -1 if antiparticle, 0 if not matched
  /// \param depthMax  maximum decay tree level to check; Daughters up to this level will be considered. If -1, all levels are considered.
  /// \return true if PDG codes of the particle and its daughters are correct, false otherwise
  template <std::size_t N, typename T, typename U>
  static bool isMatchedMCGen(const T& particlesMC, const U& candidate, int PDGParticle, array<int, N> arrPDGDaughters, bool acceptAntiParticles = false, int8_t* sign = nullptr, int depthMax = 1)
  {
    //Printf("MC Gen: Expected particle PDG: %d", PDGParticle);
    int8_t sgn = 0; // 1 if the expected mother is particle, -1 if antiparticle (w.r.t. PDGParticle)
    if (sign) {
      *sign = sgn;
    }
    // Check the PDG code of the particle.
    auto PDGCandidate = candidate.pdgCode();
    //Printf("MC Gen: Candidate PDG: %d", PDGCandidate);
    if (PDGCandidate == PDGParticle) { // exact PDG match
      sgn = 1;
    } else if (acceptAntiParticles && PDGCandidate == -PDGParticle) { // antiparticle PDG match
      sgn = -1;
    }
    if (sgn == 0) {
      //Printf("MC Gen: Rejected: bad particle PDG: %s%d != %d", acceptAntiParticles ? "abs " : "", PDGCandidate, std::abs(PDGParticle));
      return false;
    }
    // Check the PDG codes of the decay products.
    if (N > 0) {
      //Printf("MC Gen: Checking %d daughters", N);
      std::vector<int> arrAllDaughtersIndex;          // vector of indices of all daughters
      int indexDaughterFirst = candidate.daughter0(); // index of the first direct daughter
      int indexDaughterLast = candidate.daughter1();  // index of the last direct daughter
      // Check the daughter indices.
      if (indexDaughterFirst <= -1 && indexDaughterLast <= -1) {
        //Printf("MC Gen: Rejected: bad daughter index range: %d-%d", indexDaughterFirst, indexDaughterLast);
        return false;
      }
      // Check that the number of direct daughters is not larger than the number of expected final daughters.
      if (indexDaughterFirst > -1 && indexDaughterLast > -1 && indexDaughterLast - indexDaughterFirst + 1 > N) {
        //Printf("MC Gen: Rejected: too many direct daughters: %d (expected %d final)", indexDaughterLast - indexDaughterFirst + 1, N);
        return false;
      }
      // Get the list of actual final daughters.
      getDaughters(particlesMC, candidate.globalIndex(), &arrAllDaughtersIndex, arrPDGDaughters, depthMax);
      //printf("MC Gen: Mother %d has %d final daughters:", candidate.globalIndex(), arrAllDaughtersIndex.size());
      //for (auto i : arrAllDaughtersIndex) {
      //  printf(" %d", i);
      //}
      //printf("\n");
      // Check whether the number of final daughters is equal to the required number.
      if (arrAllDaughtersIndex.size() != N) {
        //Printf("MC Gen: Rejected: incorrect number of final daughters: %d (expected %d)", arrAllDaughtersIndex.size(), N);
        return false;
      }
      // Check daughters' PDG codes.
      for (auto indexDaughterI : arrAllDaughtersIndex) {
        auto candidateDaughterI = particlesMC.iteratorAt(indexDaughterI); // ith daughter particle
        auto PDGCandidateDaughterI = candidateDaughterI.pdgCode();        // PDG code of the ith daughter
        //Printf("MC Gen: Daughter %d PDG: %d", indexDaughterI, PDGCandidateDaughterI);
        bool isPDGFound = false; // Is the PDG code of this daughter among the remaining expected PDG codes?
        for (auto iProngCp = 0; iProngCp < N; ++iProngCp) {
          if (PDGCandidateDaughterI == sgn * arrPDGDaughters[iProngCp]) {
            arrPDGDaughters[iProngCp] = 0; // Remove this PDG code from the array of expected ones.
            isPDGFound = true;
            break;
          }
        }
        if (!isPDGFound) {
          //Printf("MC Gen: Rejected: bad daughter PDG: %d", PDGCandidateDaughterI);
          return false;
        }
      }
    }
    //Printf("MC Gen: Accepted: m: %d", candidate.globalIndex());
    if (sign) {
      *sign = sgn;
    }
    return true;
  }

 private:
  static std::vector<std::tuple<int, double>> mListMass; ///< list of particle masses in form (PDG code, mass)
};

std::vector<std::tuple<int, double>> RecoDecay::mListMass;

#endif // O2_ANALYSIS_RECODECAY_H_
