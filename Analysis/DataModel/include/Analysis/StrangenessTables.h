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

namespace o2::aod
{
namespace v0data
{
DECLARE_SOA_COLUMN(DCANegToPV, DCANegToPVs, float);
DECLARE_SOA_COLUMN(DCAPosToPV, DCAPosToPVs, float);
DECLARE_SOA_COLUMN(V0Radius, V0Radii, float);
DECLARE_SOA_COLUMN(DCAV0Daughter, DCAV0Daughters, float);
DECLARE_SOA_COLUMN(V0CosPA, V0CosPAs, float);
DECLARE_SOA_COLUMN(MassAsLambda, MassAsLambdas, float);
DECLARE_SOA_COLUMN(MassAsAntiLambda, MassAsAntiLambdas, float);
DECLARE_SOA_COLUMN(MassAsK0Short, MassAsK0Shorts, float);
DECLARE_SOA_COLUMN(Pt, Pts, float);
} // namespace v0data

DECLARE_SOA_TABLE(V0Data, "AOD", "V0DATA",
                  v0data::DCANegToPV, v0data::DCAPosToPV, v0data::V0Radius,
                  v0data::DCAV0Daughter, v0data::V0CosPA, v0data::MassAsLambda,
                  v0data::MassAsAntiLambda, v0data::MassAsK0Short, v0data::Pt);

namespace cascdata
{
DECLARE_SOA_COLUMN(DCANegToPV, DCANegToPVs, float);
DECLARE_SOA_COLUMN(DCAPosToPV, DCAPosToPVs, float);
DECLARE_SOA_COLUMN(DCABachToPV, DCABachToPVs, float);
DECLARE_SOA_COLUMN(V0Radius, V0Radii, float);
DECLARE_SOA_COLUMN(CascRadius, CascRadii, float);
DECLARE_SOA_COLUMN(DCAV0Daughter, DCAV0Daughters, float);
DECLARE_SOA_COLUMN(DCACascDaughter, DCACascDaughters, float);
DECLARE_SOA_COLUMN(DCAV0ToPV, DCAV0ToPVs, float);
DECLARE_SOA_COLUMN(V0CosPA, V0CosPAs, float);
DECLARE_SOA_COLUMN(CascCosPA, CascCosPAs, float);
DECLARE_SOA_COLUMN(LambdaMass, LambdaMasses, float);
DECLARE_SOA_COLUMN(MassAsXi, MassAsXis, float);
DECLARE_SOA_COLUMN(MassAsOmega, MassAsOmegas, float);
DECLARE_SOA_COLUMN(Charge, Charges, int);
DECLARE_SOA_COLUMN(Pt, Pts, float);
} // namespace cascdata

DECLARE_SOA_TABLE(CascData, "AOD", "CASCDATA",
                  cascdata::DCANegToPV, cascdata::DCAPosToPV, cascdata::DCABachToPV,
                  cascdata::V0Radius, cascdata::CascRadius, cascdata::DCAV0Daughter,
                  cascdata::DCACascDaughter, cascdata::DCAV0ToPV, cascdata::V0CosPA,
                  cascdata::CascCosPA, cascdata::LambdaMass, cascdata::MassAsXi,
                  cascdata::MassAsOmega, cascdata::Charge, cascdata::Pt);

} // namespace o2::aod

#endif // O2_ANALYSIS_STRANGENESSTABLES_H_
