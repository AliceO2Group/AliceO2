// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_STRANGENESSTABLES_H_
#define O2_ANALYSIS_STRANGENESSTABLES_H_

#include "Framework/AnalysisDataModel.h"
#include "AnalysisCore/RecoDecay.h"
#include <cmath>

namespace o2::aod
{
namespace v0data
{
//Needed to have shorter table that does not rely on existing one (filtering!)
DECLARE_SOA_INDEX_COLUMN_FULL(PosTrack, posTrack, int, Tracks, "_Pos"); //!
DECLARE_SOA_INDEX_COLUMN_FULL(NegTrack, negTrack, int, Tracks, "_Neg"); //!
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                         //!

//General V0 properties: position, momentum
DECLARE_SOA_COLUMN(PosX, posX, float);   //!
DECLARE_SOA_COLUMN(NegX, negX, float);   //!
DECLARE_SOA_COLUMN(PxPos, pxpos, float); //!
DECLARE_SOA_COLUMN(PyPos, pypos, float); //!
DECLARE_SOA_COLUMN(PzPos, pzpos, float); //!
DECLARE_SOA_COLUMN(PxNeg, pxneg, float); //!
DECLARE_SOA_COLUMN(PyNeg, pyneg, float); //!
DECLARE_SOA_COLUMN(PzNeg, pzneg, float); //!
DECLARE_SOA_COLUMN(X, x, float);         //!
DECLARE_SOA_COLUMN(Y, y, float);         //!
DECLARE_SOA_COLUMN(Z, z, float);         //!

//Saved from finding: DCAs
DECLARE_SOA_COLUMN(DCAV0Daughters, dcaV0daughters, float); //!
DECLARE_SOA_COLUMN(DCAPosToPV, dcapostopv, float);         //!
DECLARE_SOA_COLUMN(DCANegToPV, dcanegtopv, float);         //!

//Derived expressions
//Momenta
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, //!
                           [](float pxpos, float pypos, float pxneg, float pyneg) -> float { return RecoDecay::sqrtSumOfSquares(pxpos + pxneg, pypos + pyneg); });

//Length quantities
DECLARE_SOA_DYNAMIC_COLUMN(V0Radius, v0radius, //!
                           [](float x, float y) -> float { return RecoDecay::sqrtSumOfSquares(x, y); });

//CosPA
DECLARE_SOA_DYNAMIC_COLUMN(V0CosPA, v0cosPA, //!
                           [](float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ) -> float { return RecoDecay::CPA(array{pvX, pvY, pvZ}, array{X, Y, Z}, array{Px, Py, Pz}); });
DECLARE_SOA_DYNAMIC_COLUMN(DCAV0ToPV, dcav0topv, //!
                           [](float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ) -> float { return std::sqrt((std::pow((pvY - Y) * Pz - (pvZ - Z) * Py, 2) + std::pow((pvX - X) * Pz - (pvZ - Z) * Px, 2) + std::pow((pvX - X) * Py - (pvY - Y) * Px, 2)) / (Px * Px + Py * Py + Pz * Pz)); });

//Armenteros-Podolanski variables
DECLARE_SOA_DYNAMIC_COLUMN(Alpha, alpha, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) {
                             float momTot = RecoDecay::P(pxpos + pxneg, pypos + pyneg, pzpos + pzneg);
                             float lQlNeg = RecoDecay::dotProd(array{pxneg, pyneg, pzneg}, array{pxpos + pxneg, pypos + pyneg, pzpos + pzneg}) / momTot;
                             float lQlPos = RecoDecay::dotProd(array{pxpos, pypos, pzpos}, array{pxpos + pxneg, pypos + pyneg, pzpos + pzneg}) / momTot;
                             return (lQlPos - lQlNeg) / (lQlPos + lQlNeg); //alphav0
                           });

DECLARE_SOA_DYNAMIC_COLUMN(QtArm, qtarm, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) {
                             float momTot = RecoDecay::P2(pxpos + pxneg, pypos + pyneg, pzpos + pzneg);
                             float dp = RecoDecay::dotProd(array{pxneg, pyneg, pzneg}, array{pxpos + pxneg, pypos + pyneg, pzpos + pzneg});
                             return std::sqrt(RecoDecay::P2(pxneg, pyneg, pzneg) - dp * dp / momTot); //qtarm
                           });

// Psi pair angle: angle between the plane defined by the electron and positron momenta and the xy plane
DECLARE_SOA_DYNAMIC_COLUMN(PsiPair, psipair, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) {
                             auto clipToPM1 = [](float x) { return x < -1.f ? -1.f : (x > 1.f ? 1.f : x); };
                             float ptot2 = RecoDecay::P2(pxpos, pypos, pzpos) * RecoDecay::P2(pxneg, pyneg, pzneg);
                             float argcos = RecoDecay::dotProd(array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}) / std::sqrt(ptot2);
                             float thetaPos = std::atan2(RecoDecay::sqrtSumOfSquares(pxpos, pypos), pzpos);
                             float thetaNeg = std::atan2(RecoDecay::sqrtSumOfSquares(pxneg, pyneg), pzneg);
                             float argsin = (thetaNeg - thetaPos) / std::acos(clipToPM1(argcos));
                             return std::asin(clipToPM1(argsin));
                           });

//Calculated on the fly with mass assumption + dynamic tables
DECLARE_SOA_DYNAMIC_COLUMN(MLambda, mLambda, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) -> float { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kProton), RecoDecay::getMassPDG(kPiPlus)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MAntiLambda, mAntiLambda, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) -> float { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kProton)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MK0Short, mK0Short, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) -> float { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kPiPlus)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MGamma, mGamma, //!
                           [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) -> float { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kElectron), RecoDecay::getMassPDG(kElectron)}); });

DECLARE_SOA_DYNAMIC_COLUMN(YK0Short, yK0Short, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Y(array{Px, Py, Pz}, RecoDecay::getMassPDG(kK0)); });
DECLARE_SOA_DYNAMIC_COLUMN(YLambda, yLambda, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Y(array{Px, Py, Pz}, RecoDecay::getMassPDG(kLambda0)); });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Eta(array{Px, Py, Pz}); });
DECLARE_SOA_DYNAMIC_COLUMN(Phi, phi, //!
                           [](float Px, float Py) -> float { return RecoDecay::Phi(Px, Py); });

DECLARE_SOA_EXPRESSION_COLUMN(Px, px, //!
                              float, 1.f * aod::v0data::pxpos + 1.f * aod::v0data::pxneg);
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, //!
                              float, 1.f * aod::v0data::pypos + 1.f * aod::v0data::pyneg);
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, //!
                              float, 1.f * aod::v0data::pzpos + 1.f * aod::v0data::pzneg);
} // namespace v0data

DECLARE_SOA_TABLE_FULL(StoredV0Datas, "V0Datas", "AOD", "V0DATA", //!
                       o2::soa::Index<>, v0data::PosTrackId, v0data::NegTrackId, v0data::CollisionId,
                       v0data::PosX, v0data::NegX,
                       v0data::X, v0data::Y, v0data::Z,
                       v0data::PxPos, v0data::PyPos, v0data::PzPos,
                       v0data::PxNeg, v0data::PyNeg, v0data::PzNeg,
                       v0data::DCAV0Daughters, v0data::DCAPosToPV, v0data::DCANegToPV,

                       //Dynamic columns
                       v0data::Pt<v0data::PxPos, v0data::PyPos, v0data::PxNeg, v0data::PyNeg>,
                       v0data::V0Radius<v0data::X, v0data::Y>,
                       v0data::V0CosPA<v0data::X, v0data::Y, v0data::Z, v0data::Px, v0data::Py, v0data::Pz>,
                       v0data::DCAV0ToPV<v0data::X, v0data::Y, v0data::Z, v0data::Px, v0data::Py, v0data::Pz>,
                       v0data::Alpha<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                       v0data::QtArm<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                       v0data::PsiPair<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,

                       //Invariant masses
                       v0data::MLambda<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                       v0data::MAntiLambda<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                       v0data::MK0Short<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                       v0data::MGamma<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,

                       //Longitudinal
                       v0data::YK0Short<v0data::Px, v0data::Py, v0data::Pz>,
                       v0data::YLambda<v0data::Px, v0data::Py, v0data::Pz>,
                       v0data::Eta<v0data::Px, v0data::Py, v0data::Pz>,
                       v0data::Phi<v0data::Px, v0data::Py>);

// extended table with expression columns that can be used as arguments of dynamic columns
DECLARE_SOA_EXTENDED_TABLE_USER(V0Datas, StoredV0Datas, "V0DATAEXT", //!
                                v0data::Px, v0data::Py, v0data::Pz); // the table name has here to be the one with EXT which is not nice and under study

using V0Data = V0Datas::iterator;
namespace cascdata
{
//Necessary for full filtering functionality
DECLARE_SOA_INDEX_COLUMN(V0Data, v0Data);                           //!
DECLARE_SOA_INDEX_COLUMN_FULL(Bachelor, bachelor, int, Tracks, ""); //!
DECLARE_SOA_INDEX_COLUMN(Collision, collision);                     //!
//General V0 properties: position, momentum
DECLARE_SOA_COLUMN(Sign, sign, int);         //!
DECLARE_SOA_COLUMN(PxPos, pxpos, float);     //!
DECLARE_SOA_COLUMN(PyPos, pypos, float);     //!
DECLARE_SOA_COLUMN(PzPos, pzpos, float);     //!
DECLARE_SOA_COLUMN(PxNeg, pxneg, float);     //!
DECLARE_SOA_COLUMN(PyNeg, pyneg, float);     //!
DECLARE_SOA_COLUMN(PzNeg, pzneg, float);     //!
DECLARE_SOA_COLUMN(PxBach, pxbach, float);   //!
DECLARE_SOA_COLUMN(PyBach, pybach, float);   //!
DECLARE_SOA_COLUMN(PzBach, pzbach, float);   //!
DECLARE_SOA_COLUMN(X, x, float);             //!
DECLARE_SOA_COLUMN(Y, y, float);             //!
DECLARE_SOA_COLUMN(Z, z, float);             //!
DECLARE_SOA_COLUMN(Xlambda, xlambda, float); //!
DECLARE_SOA_COLUMN(Ylambda, ylambda, float); //!
DECLARE_SOA_COLUMN(Zlambda, zlambda, float); //!

//Saved from finding: DCAs
DECLARE_SOA_COLUMN(DCAV0Daughters, dcaV0daughters, float);     //!
DECLARE_SOA_COLUMN(DCACascDaughters, dcacascdaughters, float); //!
DECLARE_SOA_COLUMN(DCAPosToPV, dcapostopv, float);             //!
DECLARE_SOA_COLUMN(DCANegToPV, dcanegtopv, float);             //!
DECLARE_SOA_COLUMN(DCABachToPV, dcabachtopv, float);           //!

//Derived expressions
//Momenta
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, //!
                           [](float Px, float Py) -> float { return RecoDecay::sqrtSumOfSquares(Px, Py); });

//Length quantities
DECLARE_SOA_DYNAMIC_COLUMN(V0Radius, v0radius, //!
                           [](float xlambda, float ylambda) -> float { return RecoDecay::sqrtSumOfSquares(xlambda, ylambda); });
DECLARE_SOA_DYNAMIC_COLUMN(CascRadius, cascradius, //!
                           [](float x, float y) -> float { return RecoDecay::sqrtSumOfSquares(x, y); });

//CosPAs
DECLARE_SOA_DYNAMIC_COLUMN(V0CosPA, v0cosPA, //!
                           [](float Xlambda, float Ylambda, float Zlambda, float PxLambda, float PyLambda, float PzLambda, float pvX, float pvY, float pvZ) -> float { return RecoDecay::CPA(array{pvX, pvY, pvZ}, array{Xlambda, Ylambda, Zlambda}, array{PxLambda, PyLambda, PzLambda}); });
DECLARE_SOA_DYNAMIC_COLUMN(CascCosPA, casccosPA, //!
                           [](float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ) -> float { return RecoDecay::CPA(array{pvX, pvY, pvZ}, array{X, Y, Z}, array{Px, Py, Pz}); });
DECLARE_SOA_DYNAMIC_COLUMN(DCAV0ToPV, dcav0topv, //!
                           [](float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ) -> float { return std::sqrt((std::pow((pvY - Y) * Pz - (pvZ - Z) * Py, 2) + std::pow((pvX - X) * Pz - (pvZ - Z) * Px, 2) + std::pow((pvX - X) * Py - (pvY - Y) * Px, 2)) / (Px * Px + Py * Py + Pz * Pz)); });
DECLARE_SOA_DYNAMIC_COLUMN(DCACascToPV, dcacasctopv, //!
                           [](float X, float Y, float Z, float Px, float Py, float Pz, float pvX, float pvY, float pvZ) -> float { return std::sqrt((std::pow((pvY - Y) * Pz - (pvZ - Z) * Py, 2) + std::pow((pvX - X) * Pz - (pvZ - Z) * Px, 2) + std::pow((pvX - X) * Py - (pvY - Y) * Px, 2)) / (Px * Px + Py * Py + Pz * Pz)); });

//Calculated on the fly with mass assumption + dynamic tables
DECLARE_SOA_DYNAMIC_COLUMN(MLambda, mLambda, //!
                           [](int charge, float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) -> float { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, charge < 0 ? array{RecoDecay::getMassPDG(kProton), RecoDecay::getMassPDG(kPiPlus)} : array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kProton)}); });
//Calculated on the fly with mass assumption + dynamic tables

DECLARE_SOA_DYNAMIC_COLUMN(MXi, mXi, //!
                           [](float pxbach, float pybach, float pzbach, float PxLambda, float PyLambda, float PzLambda) -> float { return RecoDecay::M(array{array{pxbach, pybach, pzbach}, array{PxLambda, PyLambda, PzLambda}}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kLambda0)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MOmega, mOmega, //!
                           [](float pxbach, float pybach, float pzbach, float PxLambda, float PyLambda, float PzLambda) -> float { return RecoDecay::M(array{array{pxbach, pybach, pzbach}, array{PxLambda, PyLambda, PzLambda}}, array{RecoDecay::getMassPDG(kKPlus), RecoDecay::getMassPDG(kLambda0)}); });

DECLARE_SOA_DYNAMIC_COLUMN(YXi, yXi, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Y(array{Px, Py, Pz}, 1.32171); });
DECLARE_SOA_DYNAMIC_COLUMN(YOmega, yOmega, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Y(array{Px, Py, Pz}, 1.67245); });
DECLARE_SOA_DYNAMIC_COLUMN(Eta, eta, //!
                           [](float Px, float Py, float Pz) -> float { return RecoDecay::Eta(array{Px, Py, Pz}); });
} // namespace cascdata

namespace cascdataext
{
DECLARE_SOA_EXPRESSION_COLUMN(PxLambda, pxlambda, //!
                              float, 1.f * aod::cascdata::pxpos + 1.f * aod::cascdata::pxneg);
DECLARE_SOA_EXPRESSION_COLUMN(PyLambda, pylambda, //!
                              float, 1.f * aod::cascdata::pypos + 1.f * aod::cascdata::pyneg);
DECLARE_SOA_EXPRESSION_COLUMN(PzLambda, pzlambda, //!
                              float, 1.f * aod::cascdata::pzpos + 1.f * aod::cascdata::pzneg);
DECLARE_SOA_EXPRESSION_COLUMN(Px, px, //!
                              float, 1.f * aod::cascdata::pxpos + 1.f * aod::cascdata::pxneg + 1.f * aod::cascdata::pxbach);
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, //!
                              float, 1.f * aod::cascdata::pypos + 1.f * aod::cascdata::pyneg + 1.f * aod::cascdata::pybach);
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, //!
                              float, 1.f * aod::cascdata::pzpos + 1.f * aod::cascdata::pzneg + 1.f * aod::cascdata::pzbach);
} // namespace cascdataext

DECLARE_SOA_TABLE(CascData, "AOD", "CASCDATA", //!
                  o2::soa::Index<>, cascdata::V0DataId, cascdata::BachelorId, cascdata::CollisionId,

                  cascdata::Sign,
                  cascdata::X, cascdata::Y, cascdata::Z,
                  cascdata::Xlambda, cascdata::Ylambda, cascdata::Zlambda,
                  cascdata::PxPos, cascdata::PyPos, cascdata::PzPos,
                  cascdata::PxNeg, cascdata::PyNeg, cascdata::PzNeg,
                  cascdata::PxBach, cascdata::PyBach, cascdata::PzBach,
                  cascdata::DCAV0Daughters, cascdata::DCACascDaughters,
                  cascdata::DCAPosToPV, cascdata::DCANegToPV, cascdata::DCABachToPV,

                  //Dynamic columns
                  cascdata::Pt<cascdataext::Px, cascdataext::Py>,
                  cascdata::V0Radius<cascdata::Xlambda, cascdata::Ylambda>,
                  cascdata::CascRadius<cascdata::X, cascdata::Y>,
                  cascdata::V0CosPA<cascdata::Xlambda, cascdata::Ylambda, cascdata::Zlambda, cascdataext::PxLambda, cascdataext::PyLambda, cascdataext::PzLambda>,
                  cascdata::CascCosPA<cascdata::X, cascdata::Y, cascdata::Z, cascdataext::Px, cascdataext::Py, cascdataext::Pz>,
                  cascdata::DCAV0ToPV<cascdata::Xlambda, cascdata::Ylambda, cascdata::Zlambda, cascdataext::PxLambda, cascdataext::PyLambda, cascdataext::PzLambda>,
                  cascdata::DCACascToPV<cascdata::X, cascdata::Y, cascdata::Z, cascdataext::Px, cascdataext::Py, cascdataext::Pz>,

                  //Invariant masses
                  cascdata::MLambda<cascdata::Sign, cascdata::PxPos, cascdata::PyPos, cascdata::PzPos, cascdata::PxNeg, cascdata::PyNeg, cascdata::PzNeg>,
                  cascdata::MXi<cascdata::PxBach, cascdata::PyBach, cascdata::PzBach, cascdataext::PxLambda, cascdataext::PyLambda, cascdataext::PzLambda>,
                  cascdata::MOmega<cascdata::PxBach, cascdata::PyBach, cascdata::PzBach, cascdataext::PxLambda, cascdataext::PyLambda, cascdataext::PzLambda>,
                  //Longitudinal
                  cascdata::YXi<cascdataext::Px, cascdataext::Py, cascdataext::Pz>,
                  cascdata::YOmega<cascdataext::Px, cascdataext::Py, cascdataext::Pz>,
                  cascdata::Eta<cascdataext::Px, cascdataext::Py, cascdataext::Pz>);

using CascDataOrigin = CascData;

// extended table with expression columns that can be used as arguments of dynamic columns
DECLARE_SOA_EXTENDED_TABLE_USER(CascDataExt, CascDataOrigin, "CascDATAEXT", //!
                                cascdataext::PxLambda, cascdataext::PyLambda, cascdataext::PzLambda,
                                cascdataext::Px, cascdataext::Py, cascdataext::Pz);

using CascDataFull = CascDataExt;
} // namespace o2::aod

#endif // O2_ANALYSIS_STRANGENESSTABLES_H_
