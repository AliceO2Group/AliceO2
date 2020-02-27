// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef O2_ANALYSIS_SECONDARYVERTEX_H_
#define O2_ANALYSIS_SECONDARYVERTEX_H_

#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace secvtx2prong
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fCollisionsID");
DECLARE_SOA_COLUMN(Posdecayx, posdecayx, float, "fPosdecayx");
DECLARE_SOA_COLUMN(Posdecayy, posdecayy, float, "fPosdecayy");
DECLARE_SOA_COLUMN(Posdecayz, posdecayz, float, "fPosdecayz");
DECLARE_SOA_COLUMN(Index0, index0, int, "fIndex0");
DECLARE_SOA_COLUMN(Px0, px0, float, "fPx0");
DECLARE_SOA_COLUMN(Py0, py0, float, "fPy0");
DECLARE_SOA_COLUMN(Pz0, pz0, float, "fPz0");
DECLARE_SOA_COLUMN(Index1, index1, int, "fIndex1");
DECLARE_SOA_COLUMN(Px1, px1, float, "fPx1");
DECLARE_SOA_COLUMN(Py1, py1, float, "fPy1");
DECLARE_SOA_COLUMN(Pz1, pz1, float, "fPz1");
DECLARE_SOA_COLUMN(IndexDCApair, indexDCApair, int, "fIndexDCApair");
DECLARE_SOA_COLUMN(Mass, mass, float, "fMass");
DECLARE_SOA_COLUMN(Massbar, massbar, float, "fMassbar");
DECLARE_SOA_DYNAMIC_COLUMN(DecaylengthXY, decaylengthXY, [](float xvtxd, float yvtxd, float xvtxp, float yvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp)); });
DECLARE_SOA_DYNAMIC_COLUMN(Decaylength, decaylength, [](float xvtxd, float yvtxd, float zvtxd, float xvtxp, float yvtxp, float zvtxp) { return sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp) + (zvtxd - zvtxp) * (zvtxd - zvtxp)); });

//old way of doing it
//DECLARE_SOA_COLUMN(Decaylength, decaylength, float, "fDecaylength");
//DECLARE_SOA_COLUMN(DecaylengthXY, decaylengthXY, float, "fDecaylengthXY");

} // namespace secvtx2prong
namespace cand2prong
{
DECLARE_SOA_COLUMN(CollisionId, collisionId, int, "fCollisionsID");
DECLARE_SOA_COLUMN(MassD0, massD0, float, "fMassD0");
DECLARE_SOA_COLUMN(MassD0bar, massD0bar, float, "fMassD0bar");
} // namespace cand2prong

DECLARE_SOA_TABLE(SecVtx2Prong, "AOD", "CAND2PRONG",
                  secvtx2prong::CollisionId, collision::PosX, collision::PosY, collision::PosZ,
                  secvtx2prong::Posdecayx, secvtx2prong::Posdecayy, secvtx2prong::Posdecayz,
                  secvtx2prong::Index0, secvtx2prong::Px0, secvtx2prong::Py0, secvtx2prong::Pz0,
                  secvtx2prong::Index1, secvtx2prong::Px1, secvtx2prong::Py1, secvtx2prong::Pz1,
                  secvtx2prong::IndexDCApair, secvtx2prong::Mass, secvtx2prong::Massbar,
                  secvtx2prong::DecaylengthXY<secvtx2prong::Posdecayx, secvtx2prong::Posdecayy, collision::PosX, collision::PosY>,
                  secvtx2prong::Decaylength<secvtx2prong::Posdecayx, secvtx2prong::Posdecayy, secvtx2prong::Posdecayz, collision::PosX, collision::PosY, collision::PosZ>);

DECLARE_SOA_TABLE(Cand2Prong, "AOD", "CANDDZERO",
                  cand2prong::CollisionId, cand2prong::MassD0, cand2prong::MassD0bar);
} // namespace o2::aod

using namespace o2;
using namespace o2::framework;

float decaylengthXY(float xvtxp, float yvtxp, float xvtxd, float yvtxd)
{
  float decl_ = sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp));
  return decl_;
};

float decaylength(float xvtxp, float yvtxp, float zvtxp, float xvtxd, float yvtxd, float zvtxd)
{
  float decl_ = sqrtf((yvtxd - yvtxp) * (yvtxd - yvtxp) + (xvtxd - xvtxp) * (xvtxd - xvtxp) + (zvtxd - zvtxp) * (zvtxd - zvtxp));
  return decl_;
};

float energy(float px, float py, float pz, float mass)
{
  float en_ = sqrtf(mass * mass + px * px + py * py + pz * pz);
  return en_;
};

float invmass2prongs(float px0, float py0, float pz0, float mass0,
                     float px1, float py1, float pz1, float mass1)
{

  float energy0_ = energy(px0, py0, pz0, mass0);
  float energy1_ = energy(px1, py1, pz1, mass1);
  float energytot = energy0_ + energy1_;

  float psum2 = (px0 + px1) * (px0 + px1) + (py0 + py1) * (py0 + py1) + (pz0 + pz1) * (pz0 + pz1);
  float mass = sqrtf(energytot * energytot - psum2);
  return mass;
};

#endif // O2_ANALYSIS_SECONDARYVERTEX_H_
