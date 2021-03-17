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
#include "TMath.h"
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

// Macro to convert the stored Nsigmas to floats
#define DEFINE_UNWRAP_NSIGMA_COLUMN(COLUMN, COLUMN_NAME) \
  DECLARE_SOA_DYNAMIC_COLUMN(COLUMN, COLUMN_NAME,        \
                             [](binned_nsigma_t nsigma_binned) -> float { return bin_width * static_cast<float>(nsigma_binned); });

namespace pidtof_tiny
{
typedef int8_t binned_nsigma_t;
constexpr int nbins = (1 << 8 * sizeof(binned_nsigma_t)) - 2;
constexpr binned_nsigma_t upper_bin = nbins >> 1;
constexpr binned_nsigma_t lower_bin = -(nbins >> 1);
constexpr float binned_max = 6.35;
constexpr float binned_min = -6.35;
constexpr float bin_width = (binned_max - binned_min) / nbins;
// NSigma with reduced size 8 bit
DECLARE_SOA_COLUMN(TOFNSigmaStoreEl, tofNSigmaStoreEl, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreMu, tofNSigmaStoreMu, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStorePi, tofNSigmaStorePi, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreKa, tofNSigmaStoreKa, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStorePr, tofNSigmaStorePr, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreDe, tofNSigmaStoreDe, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreTr, tofNSigmaStoreTr, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreHe, tofNSigmaStoreHe, binned_nsigma_t);
DECLARE_SOA_COLUMN(TOFNSigmaStoreAl, tofNSigmaStoreAl, binned_nsigma_t);
// NSigma with reduced size in [binned_min, binned_max] bin size bin_width
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaEl, tofNSigmaEl);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaMu, tofNSigmaMu);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaPi, tofNSigmaPi);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaKa, tofNSigmaKa);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaPr, tofNSigmaPr);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaDe, tofNSigmaDe);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaTr, tofNSigmaTr);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaHe, tofNSigmaHe);
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaAl, tofNSigmaAl);

} // namespace pidtof_tiny

DECLARE_SOA_TABLE(pidRespTOFbeta, "AOD", "pidRespTOFbeta",
                  pidtofbeta::Beta, pidtofbeta::BetaError,
                  pidtofbeta::ExpBetaEl, pidtofbeta::ExpBetaElError,
                  pidtofbeta::SeparationBetaEl,
                  pidtofbeta::DiffBetaEl<pidtofbeta::Beta, pidtofbeta::ExpBetaEl>);

// Table with the full information for all particles
DECLARE_SOA_TABLE(pidRespTOF, "AOD", "pidRespTOF",
                  // Expected signals
                  pidtof::TOFExpSignalDiffEl<pidtof::TOFNSigmaEl, pidtof::TOFExpSigmaEl>,
                  pidtof::TOFExpSignalDiffMu<pidtof::TOFNSigmaMu, pidtof::TOFExpSigmaMu>,
                  pidtof::TOFExpSignalDiffPi<pidtof::TOFNSigmaPi, pidtof::TOFExpSigmaPi>,
                  pidtof::TOFExpSignalDiffKa<pidtof::TOFNSigmaKa, pidtof::TOFExpSigmaKa>,
                  pidtof::TOFExpSignalDiffPr<pidtof::TOFNSigmaPr, pidtof::TOFExpSigmaPr>,
                  pidtof::TOFExpSignalDiffDe<pidtof::TOFNSigmaDe, pidtof::TOFExpSigmaDe>,
                  pidtof::TOFExpSignalDiffTr<pidtof::TOFNSigmaTr, pidtof::TOFExpSigmaTr>,
                  pidtof::TOFExpSignalDiffHe<pidtof::TOFNSigmaHe, pidtof::TOFExpSigmaHe>,
                  pidtof::TOFExpSignalDiffAl<pidtof::TOFNSigmaAl, pidtof::TOFExpSigmaAl>,
                  // Expected sigma
                  pidtof::TOFExpSigmaEl, pidtof::TOFExpSigmaMu, pidtof::TOFExpSigmaPi,
                  pidtof::TOFExpSigmaKa, pidtof::TOFExpSigmaPr, pidtof::TOFExpSigmaDe,
                  pidtof::TOFExpSigmaTr, pidtof::TOFExpSigmaHe, pidtof::TOFExpSigmaAl,
                  // NSigma
                  pidtof::TOFNSigmaEl, pidtof::TOFNSigmaMu, pidtof::TOFNSigmaPi,
                  pidtof::TOFNSigmaKa, pidtof::TOFNSigmaPr, pidtof::TOFNSigmaDe,
                  pidtof::TOFNSigmaTr, pidtof::TOFNSigmaHe, pidtof::TOFNSigmaAl);

// Per particle tables
DECLARE_SOA_TABLE(pidRespTOFEl, "AOD", "pidRespTOFEl", pidtof::TOFExpSignalDiffEl<pidtof::TOFNSigmaEl, pidtof::TOFExpSigmaEl>, pidtof::TOFExpSigmaEl, pidtof::TOFNSigmaEl);
DECLARE_SOA_TABLE(pidRespTOFMu, "AOD", "pidRespTOFMu", pidtof::TOFExpSignalDiffMu<pidtof::TOFNSigmaMu, pidtof::TOFExpSigmaMu>, pidtof::TOFExpSigmaMu, pidtof::TOFNSigmaMu);
DECLARE_SOA_TABLE(pidRespTOFPi, "AOD", "pidRespTOFPi", pidtof::TOFExpSignalDiffPi<pidtof::TOFNSigmaPi, pidtof::TOFExpSigmaPi>, pidtof::TOFExpSigmaPi, pidtof::TOFNSigmaPi);
DECLARE_SOA_TABLE(pidRespTOFKa, "AOD", "pidRespTOFKa", pidtof::TOFExpSignalDiffKa<pidtof::TOFNSigmaKa, pidtof::TOFExpSigmaKa>, pidtof::TOFExpSigmaKa, pidtof::TOFNSigmaKa);
DECLARE_SOA_TABLE(pidRespTOFPr, "AOD", "pidRespTOFPr", pidtof::TOFExpSignalDiffPr<pidtof::TOFNSigmaPr, pidtof::TOFExpSigmaPr>, pidtof::TOFExpSigmaPr, pidtof::TOFNSigmaPr);
DECLARE_SOA_TABLE(pidRespTOFDe, "AOD", "pidRespTOFDe", pidtof::TOFExpSignalDiffDe<pidtof::TOFNSigmaDe, pidtof::TOFExpSigmaDe>, pidtof::TOFExpSigmaDe, pidtof::TOFNSigmaDe);
DECLARE_SOA_TABLE(pidRespTOFTr, "AOD", "pidRespTOFTr", pidtof::TOFExpSignalDiffTr<pidtof::TOFNSigmaTr, pidtof::TOFExpSigmaTr>, pidtof::TOFExpSigmaTr, pidtof::TOFNSigmaTr);
DECLARE_SOA_TABLE(pidRespTOFHe, "AOD", "pidRespTOFHe", pidtof::TOFExpSignalDiffHe<pidtof::TOFNSigmaHe, pidtof::TOFExpSigmaHe>, pidtof::TOFExpSigmaHe, pidtof::TOFNSigmaHe);
DECLARE_SOA_TABLE(pidRespTOFAl, "AOD", "pidRespTOFAl", pidtof::TOFExpSignalDiffAl<pidtof::TOFNSigmaAl, pidtof::TOFExpSigmaAl>, pidtof::TOFExpSigmaAl, pidtof::TOFNSigmaAl);

// Tiny size tables
DECLARE_SOA_TABLE(pidRespTOFTEl, "AOD", "pidRespTOFTEl", pidtof_tiny::TOFNSigmaStoreEl, pidtof_tiny::TOFNSigmaEl<pidtof_tiny::TOFNSigmaStoreEl>);
DECLARE_SOA_TABLE(pidRespTOFTMu, "AOD", "pidRespTOFTMu", pidtof_tiny::TOFNSigmaStoreMu, pidtof_tiny::TOFNSigmaMu<pidtof_tiny::TOFNSigmaStoreMu>);
DECLARE_SOA_TABLE(pidRespTOFTPi, "AOD", "pidRespTOFTPi", pidtof_tiny::TOFNSigmaStorePi, pidtof_tiny::TOFNSigmaPi<pidtof_tiny::TOFNSigmaStorePi>);
DECLARE_SOA_TABLE(pidRespTOFTKa, "AOD", "pidRespTOFTKa", pidtof_tiny::TOFNSigmaStoreKa, pidtof_tiny::TOFNSigmaKa<pidtof_tiny::TOFNSigmaStoreKa>);
DECLARE_SOA_TABLE(pidRespTOFTPr, "AOD", "pidRespTOFTPr", pidtof_tiny::TOFNSigmaStorePr, pidtof_tiny::TOFNSigmaPr<pidtof_tiny::TOFNSigmaStorePr>);
DECLARE_SOA_TABLE(pidRespTOFTDe, "AOD", "pidRespTOFTDe", pidtof_tiny::TOFNSigmaStoreDe, pidtof_tiny::TOFNSigmaDe<pidtof_tiny::TOFNSigmaStoreDe>);
DECLARE_SOA_TABLE(pidRespTOFTTr, "AOD", "pidRespTOFTTr", pidtof_tiny::TOFNSigmaStoreTr, pidtof_tiny::TOFNSigmaTr<pidtof_tiny::TOFNSigmaStoreTr>);
DECLARE_SOA_TABLE(pidRespTOFTHe, "AOD", "pidRespTOFTHe", pidtof_tiny::TOFNSigmaStoreHe, pidtof_tiny::TOFNSigmaHe<pidtof_tiny::TOFNSigmaStoreHe>);
DECLARE_SOA_TABLE(pidRespTOFTAl, "AOD", "pidRespTOFTAl", pidtof_tiny::TOFNSigmaStoreAl, pidtof_tiny::TOFNSigmaAl<pidtof_tiny::TOFNSigmaStoreAl>);

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

namespace pidtpc_tiny
{
typedef int8_t binned_nsigma_t;
constexpr int nbins = (1 << 8 * sizeof(binned_nsigma_t)) - 2;
constexpr binned_nsigma_t upper_bin = nbins >> 1;
constexpr binned_nsigma_t lower_bin = -(nbins >> 1);
constexpr float binned_max = 6.35;
constexpr float binned_min = -6.35;
constexpr float bin_width = (binned_max - binned_min) / nbins;
// NSigma with reduced size
DECLARE_SOA_COLUMN(TPCNSigmaStoreEl, tpcNSigmaStoreEl, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreMu, tpcNSigmaStoreMu, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStorePi, tpcNSigmaStorePi, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreKa, tpcNSigmaStoreKa, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStorePr, tpcNSigmaStorePr, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreDe, tpcNSigmaStoreDe, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreTr, tpcNSigmaStoreTr, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreHe, tpcNSigmaStoreHe, binned_nsigma_t);
DECLARE_SOA_COLUMN(TPCNSigmaStoreAl, tpcNSigmaStoreAl, binned_nsigma_t);
// NSigma with reduced size in [binned_min, binned_max] bin size bin_width
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaEl, tpcNSigmaEl);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaMu, tpcNSigmaMu);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaPi, tpcNSigmaPi);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaKa, tpcNSigmaKa);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaPr, tpcNSigmaPr);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaDe, tpcNSigmaDe);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaTr, tpcNSigmaTr);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaHe, tpcNSigmaHe);
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaAl, tpcNSigmaAl);

} // namespace pidtpc_tiny

// Table with the full information for all particles
DECLARE_SOA_TABLE(pidRespTPC, "AOD", "pidRespTPC",
                  // Expected signals
                  pidtpc::TPCExpSignalDiffEl<pidtpc::TPCNSigmaEl, pidtpc::TPCExpSigmaEl>,
                  pidtpc::TPCExpSignalDiffMu<pidtpc::TPCNSigmaMu, pidtpc::TPCExpSigmaMu>,
                  pidtpc::TPCExpSignalDiffPi<pidtpc::TPCNSigmaPi, pidtpc::TPCExpSigmaPi>,
                  pidtpc::TPCExpSignalDiffKa<pidtpc::TPCNSigmaKa, pidtpc::TPCExpSigmaKa>,
                  pidtpc::TPCExpSignalDiffPr<pidtpc::TPCNSigmaPr, pidtpc::TPCExpSigmaPr>,
                  pidtpc::TPCExpSignalDiffDe<pidtpc::TPCNSigmaDe, pidtpc::TPCExpSigmaDe>,
                  pidtpc::TPCExpSignalDiffTr<pidtpc::TPCNSigmaTr, pidtpc::TPCExpSigmaTr>,
                  pidtpc::TPCExpSignalDiffHe<pidtpc::TPCNSigmaHe, pidtpc::TPCExpSigmaHe>,
                  pidtpc::TPCExpSignalDiffAl<pidtpc::TPCNSigmaAl, pidtpc::TPCExpSigmaAl>,
                  // Expected sigma
                  pidtpc::TPCExpSigmaEl, pidtpc::TPCExpSigmaMu, pidtpc::TPCExpSigmaPi,
                  pidtpc::TPCExpSigmaKa, pidtpc::TPCExpSigmaPr, pidtpc::TPCExpSigmaDe,
                  pidtpc::TPCExpSigmaTr, pidtpc::TPCExpSigmaHe, pidtpc::TPCExpSigmaAl,
                  // NSigma
                  pidtpc::TPCNSigmaEl, pidtpc::TPCNSigmaMu, pidtpc::TPCNSigmaPi,
                  pidtpc::TPCNSigmaKa, pidtpc::TPCNSigmaPr, pidtpc::TPCNSigmaDe,
                  pidtpc::TPCNSigmaTr, pidtpc::TPCNSigmaHe, pidtpc::TPCNSigmaAl);

// Per particle tables
DECLARE_SOA_TABLE(pidRespTPCEl, "AOD", "pidRespTPCEl", pidtpc::TPCExpSignalDiffEl<pidtpc::TPCNSigmaEl, pidtpc::TPCExpSigmaEl>, pidtpc::TPCExpSigmaEl, pidtpc::TPCNSigmaEl);
DECLARE_SOA_TABLE(pidRespTPCMu, "AOD", "pidRespTPCMu", pidtpc::TPCExpSignalDiffMu<pidtpc::TPCNSigmaMu, pidtpc::TPCExpSigmaMu>, pidtpc::TPCExpSigmaMu, pidtpc::TPCNSigmaMu);
DECLARE_SOA_TABLE(pidRespTPCPi, "AOD", "pidRespTPCPi", pidtpc::TPCExpSignalDiffPi<pidtpc::TPCNSigmaPi, pidtpc::TPCExpSigmaPi>, pidtpc::TPCExpSigmaPi, pidtpc::TPCNSigmaPi);
DECLARE_SOA_TABLE(pidRespTPCKa, "AOD", "pidRespTPCKa", pidtpc::TPCExpSignalDiffKa<pidtpc::TPCNSigmaKa, pidtpc::TPCExpSigmaKa>, pidtpc::TPCExpSigmaKa, pidtpc::TPCNSigmaKa);
DECLARE_SOA_TABLE(pidRespTPCPr, "AOD", "pidRespTPCPr", pidtpc::TPCExpSignalDiffPr<pidtpc::TPCNSigmaPr, pidtpc::TPCExpSigmaPr>, pidtpc::TPCExpSigmaPr, pidtpc::TPCNSigmaPr);
DECLARE_SOA_TABLE(pidRespTPCDe, "AOD", "pidRespTPCDe", pidtpc::TPCExpSignalDiffDe<pidtpc::TPCNSigmaDe, pidtpc::TPCExpSigmaDe>, pidtpc::TPCExpSigmaDe, pidtpc::TPCNSigmaDe);
DECLARE_SOA_TABLE(pidRespTPCTr, "AOD", "pidRespTPCTr", pidtpc::TPCExpSignalDiffTr<pidtpc::TPCNSigmaTr, pidtpc::TPCExpSigmaTr>, pidtpc::TPCExpSigmaTr, pidtpc::TPCNSigmaTr);
DECLARE_SOA_TABLE(pidRespTPCHe, "AOD", "pidRespTPCHe", pidtpc::TPCExpSignalDiffHe<pidtpc::TPCNSigmaHe, pidtpc::TPCExpSigmaHe>, pidtpc::TPCExpSigmaHe, pidtpc::TPCNSigmaHe);
DECLARE_SOA_TABLE(pidRespTPCAl, "AOD", "pidRespTPCAl", pidtpc::TPCExpSignalDiffAl<pidtpc::TPCNSigmaAl, pidtpc::TPCExpSigmaAl>, pidtpc::TPCExpSigmaAl, pidtpc::TPCNSigmaAl);

// Tiny size tables
DECLARE_SOA_TABLE(pidRespTPCTEl, "AOD", "pidRespTPCTEl", pidtpc_tiny::TPCNSigmaStoreEl, pidtpc_tiny::TPCNSigmaEl<pidtpc_tiny::TPCNSigmaStoreEl>);
DECLARE_SOA_TABLE(pidRespTPCTMu, "AOD", "pidRespTPCTMu", pidtpc_tiny::TPCNSigmaStoreMu, pidtpc_tiny::TPCNSigmaMu<pidtpc_tiny::TPCNSigmaStoreMu>);
DECLARE_SOA_TABLE(pidRespTPCTPi, "AOD", "pidRespTPCTPi", pidtpc_tiny::TPCNSigmaStorePi, pidtpc_tiny::TPCNSigmaPi<pidtpc_tiny::TPCNSigmaStorePi>);
DECLARE_SOA_TABLE(pidRespTPCTKa, "AOD", "pidRespTPCTKa", pidtpc_tiny::TPCNSigmaStoreKa, pidtpc_tiny::TPCNSigmaKa<pidtpc_tiny::TPCNSigmaStoreKa>);
DECLARE_SOA_TABLE(pidRespTPCTPr, "AOD", "pidRespTPCTPr", pidtpc_tiny::TPCNSigmaStorePr, pidtpc_tiny::TPCNSigmaPr<pidtpc_tiny::TPCNSigmaStorePr>);
DECLARE_SOA_TABLE(pidRespTPCTDe, "AOD", "pidRespTPCTDe", pidtpc_tiny::TPCNSigmaStoreDe, pidtpc_tiny::TPCNSigmaDe<pidtpc_tiny::TPCNSigmaStoreDe>);
DECLARE_SOA_TABLE(pidRespTPCTTr, "AOD", "pidRespTPCTTr", pidtpc_tiny::TPCNSigmaStoreTr, pidtpc_tiny::TPCNSigmaTr<pidtpc_tiny::TPCNSigmaStoreTr>);
DECLARE_SOA_TABLE(pidRespTPCTHe, "AOD", "pidRespTPCTHe", pidtpc_tiny::TPCNSigmaStoreHe, pidtpc_tiny::TPCNSigmaHe<pidtpc_tiny::TPCNSigmaStoreHe>);
DECLARE_SOA_TABLE(pidRespTPCTAl, "AOD", "pidRespTPCTAl", pidtpc_tiny::TPCNSigmaStoreAl, pidtpc_tiny::TPCNSigmaAl<pidtpc_tiny::TPCNSigmaStoreAl>);

#undef DEFINE_UNWRAP_NSIGMA_COLUMN

} // namespace o2::aod

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
