// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

///
/// \file   PIDResponse.h
/// \author Nicolo' Jacazio
/// \brief Set of tables, tasks and utilities to provide the interface between
///        the analysis data model and the PID response
///

#ifndef O2_FRAMEWORK_PIDRESPONSE_H_
#define O2_FRAMEWORK_PIDRESPONSE_H_

// O2 includes
#include "Framework/ASoA.h"
#include "Framework/AnalysisTask.h"
#include "Framework/AnalysisDataModel.h"
#include "ReconstructionDataFormats/Track.h"
#include "ReconstructionDataFormats/PID.h"

namespace o2::aod
{

namespace pidtofbeta
{
DECLARE_SOA_COLUMN(Beta, beta, float);
DECLARE_SOA_COLUMN(BetaError, betaerror, float);
//
DECLARE_SOA_COLUMN(ExpBetaEl, expbetael, float);
DECLARE_SOA_COLUMN(ExpBetaElError, expbetaelerror, float);
//
DECLARE_SOA_COLUMN(SeparationBetaEl, separationbetael, float);
DECLARE_SOA_DYNAMIC_COLUMN(DiffBetaEl, diffbetael, [](float beta, float expbetael) -> float { return beta - expbetael; });
} // namespace pidtofbeta

namespace pidtof
{
// Expected times
DECLARE_SOA_COLUMN(TOFExpSignalEl, tofExpSignalEl, float);
DECLARE_SOA_COLUMN(TOFExpSignalMu, tofExpSignalMu, float);
DECLARE_SOA_COLUMN(TOFExpSignalPi, tofExpSignalPi, float);
DECLARE_SOA_COLUMN(TOFExpSignalKa, tofExpSignalKa, float);
DECLARE_SOA_COLUMN(TOFExpSignalPr, tofExpSignalPr, float);
DECLARE_SOA_COLUMN(TOFExpSignalDe, tofExpSignalDe, float);
DECLARE_SOA_COLUMN(TOFExpSignalTr, tofExpSignalTr, float);
DECLARE_SOA_COLUMN(TOFExpSignalHe, tofExpSignalHe, float);
DECLARE_SOA_COLUMN(TOFExpSignalAl, tofExpSignalAl, float);
// Expected sigma
DECLARE_SOA_COLUMN(TOFExpSigmaEl, tofExpSigmaEl, float);
DECLARE_SOA_COLUMN(TOFExpSigmaMu, tofExpSigmaMu, float);
DECLARE_SOA_COLUMN(TOFExpSigmaPi, tofExpSigmaPi, float);
DECLARE_SOA_COLUMN(TOFExpSigmaKa, tofExpSigmaKa, float);
DECLARE_SOA_COLUMN(TOFExpSigmaPr, tofExpSigmaPr, float);
DECLARE_SOA_COLUMN(TOFExpSigmaDe, tofExpSigmaDe, float);
DECLARE_SOA_COLUMN(TOFExpSigmaTr, tofExpSigmaTr, float);
DECLARE_SOA_COLUMN(TOFExpSigmaHe, tofExpSigmaHe, float);
DECLARE_SOA_COLUMN(TOFExpSigmaAl, tofExpSigmaAl, float);
// NSigma
DECLARE_SOA_COLUMN(TOFNSigmaEl, tofNSigmaEl, float);
DECLARE_SOA_COLUMN(TOFNSigmaMu, tofNSigmaMu, float);
DECLARE_SOA_COLUMN(TOFNSigmaPi, tofNSigmaPi, float);
DECLARE_SOA_COLUMN(TOFNSigmaKa, tofNSigmaKa, float);
DECLARE_SOA_COLUMN(TOFNSigmaPr, tofNSigmaPr, float);
DECLARE_SOA_COLUMN(TOFNSigmaDe, tofNSigmaDe, float);
DECLARE_SOA_COLUMN(TOFNSigmaTr, tofNSigmaTr, float);
DECLARE_SOA_COLUMN(TOFNSigmaHe, tofNSigmaHe, float);
DECLARE_SOA_COLUMN(TOFNSigmaAl, tofNSigmaAl, float);
} // namespace pidtof

using namespace pidtofbeta;
DECLARE_SOA_TABLE(pidRespTOFbeta, "AOD", "pidRespTOFbeta",
                  Beta, BetaError,
                  ExpBetaEl, ExpBetaElError,
                  SeparationBetaEl,
                  DiffBetaEl<Beta, ExpBetaEl>);
using namespace pidtof;
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  TOFExpSignalEl, TOFExpSignalMu, TOFExpSignalPi, TOFExpSignalKa, TOFExpSignalPr, TOFExpSignalDe, TOFExpSignalTr, TOFExpSignalHe, TOFExpSignalAl,
                  TOFExpSigmaEl, TOFExpSigmaMu, TOFExpSigmaPi, TOFExpSigmaKa, TOFExpSigmaPr, TOFExpSigmaDe, TOFExpSigmaTr, TOFExpSigmaHe, TOFExpSigmaAl,
                  TOFNSigmaEl, TOFNSigmaMu, TOFNSigmaPi, TOFNSigmaKa, TOFNSigmaPr, TOFNSigmaDe, TOFNSigmaTr, TOFNSigmaHe, TOFNSigmaAl);

namespace pidtpc
{
// Expected signals
DECLARE_SOA_COLUMN(TPCExpSignalEl, tpcExpSignalEl, float);
DECLARE_SOA_COLUMN(TPCExpSignalMu, tpcExpSignalMu, float);
DECLARE_SOA_COLUMN(TPCExpSignalPi, tpcExpSignalPi, float);
DECLARE_SOA_COLUMN(TPCExpSignalKa, tpcExpSignalKa, float);
DECLARE_SOA_COLUMN(TPCExpSignalPr, tpcExpSignalPr, float);
DECLARE_SOA_COLUMN(TPCExpSignalDe, tpcExpSignalDe, float);
DECLARE_SOA_COLUMN(TPCExpSignalTr, tpcExpSignalTr, float);
DECLARE_SOA_COLUMN(TPCExpSignalHe, tpcExpSignalHe, float);
DECLARE_SOA_COLUMN(TPCExpSignalAl, tpcExpSignalAl, float);
// Expected sigma
DECLARE_SOA_COLUMN(TPCExpSigmaEl, tpcExpSigmaEl, float);
DECLARE_SOA_COLUMN(TPCExpSigmaMu, tpcExpSigmaMu, float);
DECLARE_SOA_COLUMN(TPCExpSigmaPi, tpcExpSigmaPi, float);
DECLARE_SOA_COLUMN(TPCExpSigmaKa, tpcExpSigmaKa, float);
DECLARE_SOA_COLUMN(TPCExpSigmaPr, tpcExpSigmaPr, float);
DECLARE_SOA_COLUMN(TPCExpSigmaDe, tpcExpSigmaDe, float);
DECLARE_SOA_COLUMN(TPCExpSigmaTr, tpcExpSigmaTr, float);
DECLARE_SOA_COLUMN(TPCExpSigmaHe, tpcExpSigmaHe, float);
DECLARE_SOA_COLUMN(TPCExpSigmaAl, tpcExpSigmaAl, float);
// NSigma
DECLARE_SOA_COLUMN(TPCNSigmaEl, tpcNSigmaEl, float);
DECLARE_SOA_COLUMN(TPCNSigmaMu, tpcNSigmaMu, float);
DECLARE_SOA_COLUMN(TPCNSigmaPi, tpcNSigmaPi, float);
DECLARE_SOA_COLUMN(TPCNSigmaKa, tpcNSigmaKa, float);
DECLARE_SOA_COLUMN(TPCNSigmaPr, tpcNSigmaPr, float);
DECLARE_SOA_COLUMN(TPCNSigmaDe, tpcNSigmaDe, float);
DECLARE_SOA_COLUMN(TPCNSigmaTr, tpcNSigmaTr, float);
DECLARE_SOA_COLUMN(TPCNSigmaHe, tpcNSigmaHe, float);
DECLARE_SOA_COLUMN(TPCNSigmaAl, tpcNSigmaAl, float);
} // namespace pidtpc

using namespace pidtpc;
DECLARE_SOA_TABLE(pidRespTPC, "AOD", "pidRespTPC",
                  TPCExpSignalEl, TPCExpSignalMu, TPCExpSignalPi, TPCExpSignalKa, TPCExpSignalPr, TPCExpSignalDe, TPCExpSignalTr, TPCExpSignalHe, TPCExpSignalAl,
                  TPCExpSigmaEl, TPCExpSigmaMu, TPCExpSigmaPi, TPCExpSigmaKa, TPCExpSigmaPr, TPCExpSigmaDe, TPCExpSigmaTr, TPCExpSigmaHe, TPCExpSigmaAl,
                  TPCNSigmaEl, TPCNSigmaMu, TPCNSigmaPi, TPCNSigmaKa, TPCNSigmaPr, TPCNSigmaDe, TPCNSigmaTr, TPCNSigmaHe, TPCNSigmaAl);

} // namespace o2::aod

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
