// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_STRANGENESSTABLES_H_
#define O2_ANALYSIS_STRANGENESSTABLES_H_

#include "Framework/AnalysisDataModel.h"
#include "Analysis/RecoDecay.h"

namespace o2::aod
{
namespace v0data
{
//General V0 properties: position, momentum
DECLARE_SOA_COLUMN(PxPos, pxpos, float);
DECLARE_SOA_COLUMN(PyPos, pypos, float);
DECLARE_SOA_COLUMN(PzPos, pzpos, float);
DECLARE_SOA_COLUMN(PxNeg, pxneg, float);
DECLARE_SOA_COLUMN(PyNeg, pyneg, float);
DECLARE_SOA_COLUMN(PzNeg, pzneg, float);
DECLARE_SOA_COLUMN(X, x, float);
DECLARE_SOA_COLUMN(Y, y, float);
DECLARE_SOA_COLUMN(Z, z, float);

//Saved from finding: DCAs
DECLARE_SOA_COLUMN(DCAV0Daughters, dcaV0daughters, float);
DECLARE_SOA_COLUMN(DCAPosToPV, dcapostopv, float);
DECLARE_SOA_COLUMN(DCANegToPV, dcanegtopv, float);


//Momenta
DECLARE_SOA_EXPRESSION_COLUMN(Px, px, float, 1.f*pxpos+1.f*pxneg );
DECLARE_SOA_EXPRESSION_COLUMN(Py, py, float, 1.f*pypos+1.f*pyneg );
DECLARE_SOA_EXPRESSION_COLUMN(Pz, pz, float, 1.f*pzpos+1.f*pzneg );
DECLARE_SOA_DYNAMIC_COLUMN(Pt, pt, [](float px, float py) { return RecoDecay::sqrtSumOfSquares(px,py); });

//Length quantities
DECLARE_SOA_DYNAMIC_COLUMN(V0Radius, v0radius, [](float x, float y) { return RecoDecay::sqrtSumOfSquares(x, y); });

//CosPA
DECLARE_SOA_DYNAMIC_COLUMN(V0CosPA, v0cosPA, [](float pvX, float pvY, float pvZ, float X, float Y, float Z, float Px, float Py, float Pz) { return RecoDecay::CPA(array{pvX, pvY, pvZ}, array{X, Y, Z}, array{Px, Py, Pz}); });

//Calculated on the fly with mass assumption + dynamic tables
DECLARE_SOA_DYNAMIC_COLUMN(MLambda, mLambda, [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kProton), RecoDecay::getMassPDG(kPiPlus)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MAntiLambda, mAntiLambda, [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kProton)}); });
DECLARE_SOA_DYNAMIC_COLUMN(MK0Short, mK0Short, [](float pxpos, float pypos, float pzpos, float pxneg, float pyneg, float pzneg) { return RecoDecay::M(array{array{pxpos, pypos, pzpos}, array{pxneg, pyneg, pzneg}}, array{RecoDecay::getMassPDG(kPiPlus), RecoDecay::getMassPDG(kPiPlus)}); });

} // namespace v0data

DECLARE_SOA_TABLE(V0Data, "AOD", "V0DATA",
                  v0data::X, v0data::Y, v0data::Z,
                  v0data::PxPos, v0data::PyPos, v0data::PzPos,
                  v0data::PxNeg, v0data::PyNeg, v0data::PzNeg,
                  v0data::DCAV0Daughters, v0data::DCAPosToPV, v0data::DCANegToPV,
                  
                  //Dynamic columns
                  v0data::Pt<v0data::Px,v0data::Py>,
                  v0data::V0Radius<v0data::X,v0data::Y>,
                  v0data::V0CosPA<collision::PosX, collision::PosY, collision::PosZ, v0data::X, v0data::Y, v0data::Z, v0data::Px, v0data::Py, v0data::Pz>,
                  
                  //Invariant masses
                  v0data::MLambda<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                  v0data::MAntiLambda<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>,
                  v0data::MK0Short<v0data::PxPos, v0data::PyPos, v0data::PzPos, v0data::PxNeg, v0data::PyNeg, v0data::PzNeg>
                  );

} // namespace o2::aod

#endif // O2_ANALYSIS_STRANGENESSTABLES_H_
