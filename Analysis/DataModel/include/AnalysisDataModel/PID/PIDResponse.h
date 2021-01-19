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
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffEl, tofExpSignalDiffEl, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffMu, tofExpSignalDiffMu, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffPi, tofExpSignalDiffPi, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffKa, tofExpSignalDiffKa, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffPr, tofExpSignalDiffPr, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffDe, tofExpSignalDiffDe, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffTr, tofExpSignalDiffTr, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffHe, tofExpSignalDiffHe, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffAl, tofExpSignalDiffAl, [](float nsigma, float sigma) { return nsigma * sigma; });
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
                  // Expected signals
                  TOFExpSignalDiffEl<TOFNSigmaEl, TOFExpSigmaEl>,
                  TOFExpSignalDiffMu<TOFNSigmaMu, TOFExpSigmaMu>,
                  TOFExpSignalDiffPi<TOFNSigmaPi, TOFExpSigmaPi>,
                  TOFExpSignalDiffKa<TOFNSigmaKa, TOFExpSigmaKa>,
                  TOFExpSignalDiffPr<TOFNSigmaPr, TOFExpSigmaPr>,
                  TOFExpSignalDiffDe<TOFNSigmaDe, TOFExpSigmaDe>,
                  TOFExpSignalDiffTr<TOFNSigmaTr, TOFExpSigmaTr>,
                  TOFExpSignalDiffHe<TOFNSigmaHe, TOFExpSigmaHe>,
                  TOFExpSignalDiffAl<TOFNSigmaAl, TOFExpSigmaAl>,
                  // Expected sigma
                  TOFExpSigmaEl, TOFExpSigmaMu, TOFExpSigmaPi,
                  TOFExpSigmaKa, TOFExpSigmaPr, TOFExpSigmaDe,
                  TOFExpSigmaTr, TOFExpSigmaHe, TOFExpSigmaAl,
                  // NSigma
                  TOFNSigmaEl, TOFNSigmaMu, TOFNSigmaPi,
                  TOFNSigmaKa, TOFNSigmaPr, TOFNSigmaDe,
                  TOFNSigmaTr, TOFNSigmaHe, TOFNSigmaAl);

// Per particle tables
DECLARE_SOA_TABLE(pidRespTOFEl, "AOD", "pidRespTOFEl", TOFExpSignalDiffEl<TOFNSigmaEl, TOFExpSigmaEl>, TOFExpSigmaEl, TOFNSigmaEl);
DECLARE_SOA_TABLE(pidRespTOFMu, "AOD", "pidRespTOFMu", TOFExpSignalDiffMu<TOFNSigmaMu, TOFExpSigmaMu>, TOFExpSigmaMu, TOFNSigmaMu);
DECLARE_SOA_TABLE(pidRespTOFPi, "AOD", "pidRespTOFPi", TOFExpSignalDiffPi<TOFNSigmaPi, TOFExpSigmaPi>, TOFExpSigmaPi, TOFNSigmaPi);
DECLARE_SOA_TABLE(pidRespTOFKa, "AOD", "pidRespTOFKa", TOFExpSignalDiffKa<TOFNSigmaKa, TOFExpSigmaKa>, TOFExpSigmaKa, TOFNSigmaKa);
DECLARE_SOA_TABLE(pidRespTOFPr, "AOD", "pidRespTOFPr", TOFExpSignalDiffPr<TOFNSigmaPr, TOFExpSigmaPr>, TOFExpSigmaPr, TOFNSigmaPr);
DECLARE_SOA_TABLE(pidRespTOFDe, "AOD", "pidRespTOFDe", TOFExpSignalDiffDe<TOFNSigmaDe, TOFExpSigmaDe>, TOFExpSigmaDe, TOFNSigmaDe);
DECLARE_SOA_TABLE(pidRespTOFTr, "AOD", "pidRespTOFTr", TOFExpSignalDiffTr<TOFNSigmaTr, TOFExpSigmaTr>, TOFExpSigmaTr, TOFNSigmaTr);
DECLARE_SOA_TABLE(pidRespTOFHe, "AOD", "pidRespTOFHe", TOFExpSignalDiffHe<TOFNSigmaHe, TOFExpSigmaHe>, TOFExpSigmaHe, TOFNSigmaHe);
DECLARE_SOA_TABLE(pidRespTOFAl, "AOD", "pidRespTOFAl", TOFExpSignalDiffAl<TOFNSigmaAl, TOFExpSigmaAl>, TOFExpSigmaAl, TOFNSigmaAl);

namespace pidtpc
{
// Expected signals
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffEl, tpcExpSignalDiffEl, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffMu, tpcExpSignalDiffMu, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffPi, tpcExpSignalDiffPi, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffKa, tpcExpSignalDiffKa, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffPr, tpcExpSignalDiffPr, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffDe, tpcExpSignalDiffDe, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffTr, tpcExpSignalDiffTr, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffHe, tpcExpSignalDiffHe, [](float nsigma, float sigma) { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffAl, tpcExpSignalDiffAl, [](float nsigma, float sigma) { return nsigma * sigma; });
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
                  // Expected signals
                  TPCExpSignalDiffEl<TPCNSigmaEl, TPCExpSigmaEl>,
                  TPCExpSignalDiffMu<TPCNSigmaMu, TPCExpSigmaMu>,
                  TPCExpSignalDiffPi<TPCNSigmaPi, TPCExpSigmaPi>,
                  TPCExpSignalDiffKa<TPCNSigmaKa, TPCExpSigmaKa>,
                  TPCExpSignalDiffPr<TPCNSigmaPr, TPCExpSigmaPr>,
                  TPCExpSignalDiffDe<TPCNSigmaDe, TPCExpSigmaDe>,
                  TPCExpSignalDiffTr<TPCNSigmaTr, TPCExpSigmaTr>,
                  TPCExpSignalDiffHe<TPCNSigmaHe, TPCExpSigmaHe>,
                  TPCExpSignalDiffAl<TPCNSigmaAl, TPCExpSigmaAl>,
                  // Expected sigma
                  TPCExpSigmaEl, TPCExpSigmaMu, TPCExpSigmaPi,
                  TPCExpSigmaKa, TPCExpSigmaPr, TPCExpSigmaDe,
                  TPCExpSigmaTr, TPCExpSigmaHe, TPCExpSigmaAl,
                  // NSigma
                  TPCNSigmaEl, TPCNSigmaMu, TPCNSigmaPi,
                  TPCNSigmaKa, TPCNSigmaPr, TPCNSigmaDe,
                  TPCNSigmaTr, TPCNSigmaHe, TPCNSigmaAl);

// Per particle tables
DECLARE_SOA_TABLE(pidRespTPCEl, "AOD", "pidRespTPCEl", TPCExpSignalDiffEl<TPCNSigmaEl, TPCExpSigmaEl>, TPCExpSigmaEl, TPCNSigmaEl);
DECLARE_SOA_TABLE(pidRespTPCMu, "AOD", "pidRespTPCMu", TPCExpSignalDiffMu<TPCNSigmaMu, TPCExpSigmaMu>, TPCExpSigmaMu, TPCNSigmaMu);
DECLARE_SOA_TABLE(pidRespTPCPi, "AOD", "pidRespTPCPi", TPCExpSignalDiffPi<TPCNSigmaPi, TPCExpSigmaPi>, TPCExpSigmaPi, TPCNSigmaPi);
DECLARE_SOA_TABLE(pidRespTPCKa, "AOD", "pidRespTPCKa", TPCExpSignalDiffKa<TPCNSigmaKa, TPCExpSigmaKa>, TPCExpSigmaKa, TPCNSigmaKa);
DECLARE_SOA_TABLE(pidRespTPCPr, "AOD", "pidRespTPCPr", TPCExpSignalDiffPr<TPCNSigmaPr, TPCExpSigmaPr>, TPCExpSigmaPr, TPCNSigmaPr);
DECLARE_SOA_TABLE(pidRespTPCDe, "AOD", "pidRespTPCDe", TPCExpSignalDiffDe<TPCNSigmaDe, TPCExpSigmaDe>, TPCExpSigmaDe, TPCNSigmaDe);
DECLARE_SOA_TABLE(pidRespTPCTr, "AOD", "pidRespTPCTr", TPCExpSignalDiffTr<TPCNSigmaTr, TPCExpSigmaTr>, TPCExpSigmaTr, TPCNSigmaTr);
DECLARE_SOA_TABLE(pidRespTPCHe, "AOD", "pidRespTPCHe", TPCExpSignalDiffHe<TPCNSigmaHe, TPCExpSigmaHe>, TPCExpSigmaHe, TPCNSigmaHe);
DECLARE_SOA_TABLE(pidRespTPCAl, "AOD", "pidRespTPCAl", TPCExpSignalDiffAl<TPCNSigmaAl, TPCExpSigmaAl>, TPCExpSigmaAl, TPCNSigmaAl);

} // namespace o2::aod

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
