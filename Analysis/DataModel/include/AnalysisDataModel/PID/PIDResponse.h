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
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{

namespace pidtofbeta
{
DECLARE_SOA_COLUMN(Beta, beta, float);           //! TOF beta
DECLARE_SOA_COLUMN(BetaError, betaerror, float); //! Uncertainty on the TOF beta
//
DECLARE_SOA_COLUMN(ExpBetaEl, expbetael, float);           //! Expected beta of electron
DECLARE_SOA_COLUMN(ExpBetaElError, expbetaelerror, float); //! Expected uncertainty on the beta of electron
//
DECLARE_SOA_COLUMN(SeparationBetaEl, separationbetael, float); //!
DECLARE_SOA_DYNAMIC_COLUMN(DiffBetaEl, diffbetael,             //!
                           [](float beta, float expbetael) -> float { return beta - expbetael; });
} // namespace pidtofbeta

namespace pidtof
{
// Expected times
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffEl, tofExpSignalDiffEl, //! Difference between signal and expected for electron
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffMu, tofExpSignalDiffMu, //! Difference between signal and expected for muon
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffPi, tofExpSignalDiffPi, //! Difference between signal and expected for pion
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffKa, tofExpSignalDiffKa, //! Difference between signal and expected for kaon
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffPr, tofExpSignalDiffPr, //! Difference between signal and expected for proton
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffDe, tofExpSignalDiffDe, //! Difference between signal and expected for deuteron
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffTr, tofExpSignalDiffTr, //! Difference between signal and expected for triton
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffHe, tofExpSignalDiffHe, //! Difference between signal and expected for helium3
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TOFExpSignalDiffAl, tofExpSignalDiffAl, //! Difference between signal and expected for alpha
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
// Expected sigma
DECLARE_SOA_COLUMN(TOFExpSigmaEl, tofExpSigmaEl, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaMu, tofExpSigmaMu, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaPi, tofExpSigmaPi, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaKa, tofExpSigmaKa, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaPr, tofExpSigmaPr, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaDe, tofExpSigmaDe, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaTr, tofExpSigmaTr, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaHe, tofExpSigmaHe, float); //!
DECLARE_SOA_COLUMN(TOFExpSigmaAl, tofExpSigmaAl, float); //!
// NSigma
DECLARE_SOA_COLUMN(TOFNSigmaEl, tofNSigmaEl, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaMu, tofNSigmaMu, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaPi, tofNSigmaPi, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaKa, tofNSigmaKa, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaPr, tofNSigmaPr, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaDe, tofNSigmaDe, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaTr, tofNSigmaTr, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaHe, tofNSigmaHe, float); //!
DECLARE_SOA_COLUMN(TOFNSigmaAl, tofNSigmaAl, float); //!
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
DECLARE_SOA_COLUMN(TOFNSigmaStoreEl, tofNSigmaStoreEl, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreMu, tofNSigmaStoreMu, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStorePi, tofNSigmaStorePi, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreKa, tofNSigmaStoreKa, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStorePr, tofNSigmaStorePr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreDe, tofNSigmaStoreDe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreTr, tofNSigmaStoreTr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreHe, tofNSigmaStoreHe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TOFNSigmaStoreAl, tofNSigmaStoreAl, binned_nsigma_t); //!
// NSigma with reduced size in [binned_min, binned_max] bin size bin_width
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaEl, tofNSigmaEl); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaMu, tofNSigmaMu); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaPi, tofNSigmaPi); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaKa, tofNSigmaKa); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaPr, tofNSigmaPr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaDe, tofNSigmaDe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaTr, tofNSigmaTr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaHe, tofNSigmaHe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TOFNSigmaAl, tofNSigmaAl); //!

} // namespace pidtof_tiny

DECLARE_SOA_TABLE(pidTOFbeta, "AOD", "pidTOFbeta", //! Table of the TOF beta
                  pidtofbeta::Beta, pidtofbeta::BetaError,
                  pidtofbeta::ExpBetaEl, pidtofbeta::ExpBetaElError,
                  pidtofbeta::SeparationBetaEl,
                  pidtofbeta::DiffBetaEl<pidtofbeta::Beta, pidtofbeta::ExpBetaEl>);

// Per particle tables
DECLARE_SOA_TABLE(pidTOFFullEl, "AOD", "pidTOFFullEl", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for electron
                  pidtof::TOFExpSignalDiffEl<pidtof::TOFNSigmaEl, pidtof::TOFExpSigmaEl>, pidtof::TOFExpSigmaEl, pidtof::TOFNSigmaEl);
DECLARE_SOA_TABLE(pidTOFFullMu, "AOD", "pidTOFFullMu", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for muon
                  pidtof::TOFExpSignalDiffMu<pidtof::TOFNSigmaMu, pidtof::TOFExpSigmaMu>, pidtof::TOFExpSigmaMu, pidtof::TOFNSigmaMu);
DECLARE_SOA_TABLE(pidTOFFullPi, "AOD", "pidTOFFullPi", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for pion
                  pidtof::TOFExpSignalDiffPi<pidtof::TOFNSigmaPi, pidtof::TOFExpSigmaPi>, pidtof::TOFExpSigmaPi, pidtof::TOFNSigmaPi);
DECLARE_SOA_TABLE(pidTOFFullKa, "AOD", "pidTOFFullKa", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for kaon
                  pidtof::TOFExpSignalDiffKa<pidtof::TOFNSigmaKa, pidtof::TOFExpSigmaKa>, pidtof::TOFExpSigmaKa, pidtof::TOFNSigmaKa);
DECLARE_SOA_TABLE(pidTOFFullPr, "AOD", "pidTOFFullPr", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for proton
                  pidtof::TOFExpSignalDiffPr<pidtof::TOFNSigmaPr, pidtof::TOFExpSigmaPr>, pidtof::TOFExpSigmaPr, pidtof::TOFNSigmaPr);
DECLARE_SOA_TABLE(pidTOFFullDe, "AOD", "pidTOFFullDe", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for deuteron
                  pidtof::TOFExpSignalDiffDe<pidtof::TOFNSigmaDe, pidtof::TOFExpSigmaDe>, pidtof::TOFExpSigmaDe, pidtof::TOFNSigmaDe);
DECLARE_SOA_TABLE(pidTOFFullTr, "AOD", "pidTOFFullTr", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for triton
                  pidtof::TOFExpSignalDiffTr<pidtof::TOFNSigmaTr, pidtof::TOFExpSigmaTr>, pidtof::TOFExpSigmaTr, pidtof::TOFNSigmaTr);
DECLARE_SOA_TABLE(pidTOFFullHe, "AOD", "pidTOFFullHe", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for helium3
                  pidtof::TOFExpSignalDiffHe<pidtof::TOFNSigmaHe, pidtof::TOFExpSigmaHe>, pidtof::TOFExpSigmaHe, pidtof::TOFNSigmaHe);
DECLARE_SOA_TABLE(pidTOFFullAl, "AOD", "pidTOFFullAl", //! Table of the TOF (full) response with expected signal, expected resolution and Nsigma for alpha
                  pidtof::TOFExpSignalDiffAl<pidtof::TOFNSigmaAl, pidtof::TOFExpSigmaAl>, pidtof::TOFExpSigmaAl, pidtof::TOFNSigmaAl);

// Tiny size tables
DECLARE_SOA_TABLE(pidTOFEl, "AOD", "pidTOFEl", //! Table of the TOF response with binned Nsigma for electron
                  pidtof_tiny::TOFNSigmaStoreEl, pidtof_tiny::TOFNSigmaEl<pidtof_tiny::TOFNSigmaStoreEl>);
DECLARE_SOA_TABLE(pidTOFMu, "AOD", "pidTOFMu", //! Table of the TOF response with binned Nsigma for muon
                  pidtof_tiny::TOFNSigmaStoreMu, pidtof_tiny::TOFNSigmaMu<pidtof_tiny::TOFNSigmaStoreMu>);
DECLARE_SOA_TABLE(pidTOFPi, "AOD", "pidTOFPi", //! Table of the TOF response with binned Nsigma for pion
                  pidtof_tiny::TOFNSigmaStorePi, pidtof_tiny::TOFNSigmaPi<pidtof_tiny::TOFNSigmaStorePi>);
DECLARE_SOA_TABLE(pidTOFKa, "AOD", "pidTOFKa", //! Table of the TOF response with binned Nsigma for kaon
                  pidtof_tiny::TOFNSigmaStoreKa, pidtof_tiny::TOFNSigmaKa<pidtof_tiny::TOFNSigmaStoreKa>);
DECLARE_SOA_TABLE(pidTOFPr, "AOD", "pidTOFPr", //! Table of the TOF response with binned Nsigma for proton
                  pidtof_tiny::TOFNSigmaStorePr, pidtof_tiny::TOFNSigmaPr<pidtof_tiny::TOFNSigmaStorePr>);
DECLARE_SOA_TABLE(pidTOFDe, "AOD", "pidTOFDe", //! Table of the TOF response with binned Nsigma for deuteron
                  pidtof_tiny::TOFNSigmaStoreDe, pidtof_tiny::TOFNSigmaDe<pidtof_tiny::TOFNSigmaStoreDe>);
DECLARE_SOA_TABLE(pidTOFTr, "AOD", "pidTOFTr", //! Table of the TOF response with binned Nsigma for triton
                  pidtof_tiny::TOFNSigmaStoreTr, pidtof_tiny::TOFNSigmaTr<pidtof_tiny::TOFNSigmaStoreTr>);
DECLARE_SOA_TABLE(pidTOFHe, "AOD", "pidTOFHe", //! Table of the TOF response with binned Nsigma for helium3
                  pidtof_tiny::TOFNSigmaStoreHe, pidtof_tiny::TOFNSigmaHe<pidtof_tiny::TOFNSigmaStoreHe>);
DECLARE_SOA_TABLE(pidTOFAl, "AOD", "pidTOFAl", //! Table of the TOF response with binned Nsigma for alpha
                  pidtof_tiny::TOFNSigmaStoreAl, pidtof_tiny::TOFNSigmaAl<pidtof_tiny::TOFNSigmaStoreAl>);

namespace pidtpc
{
// Expected signals
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffEl, tpcExpSignalDiffEl, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffMu, tpcExpSignalDiffMu, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffPi, tpcExpSignalDiffPi, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffKa, tpcExpSignalDiffKa, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffPr, tpcExpSignalDiffPr, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffDe, tpcExpSignalDiffDe, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffTr, tpcExpSignalDiffTr, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffHe, tpcExpSignalDiffHe, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(TPCExpSignalDiffAl, tpcExpSignalDiffAl, //!
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
// Expected sigma
DECLARE_SOA_COLUMN(TPCExpSigmaEl, tpcExpSigmaEl, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaMu, tpcExpSigmaMu, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaPi, tpcExpSigmaPi, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaKa, tpcExpSigmaKa, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaPr, tpcExpSigmaPr, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaDe, tpcExpSigmaDe, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaTr, tpcExpSigmaTr, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaHe, tpcExpSigmaHe, float); //!
DECLARE_SOA_COLUMN(TPCExpSigmaAl, tpcExpSigmaAl, float); //!
// NSigma
DECLARE_SOA_COLUMN(TPCNSigmaEl, tpcNSigmaEl, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaMu, tpcNSigmaMu, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaPi, tpcNSigmaPi, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaKa, tpcNSigmaKa, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaPr, tpcNSigmaPr, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaDe, tpcNSigmaDe, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaTr, tpcNSigmaTr, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaHe, tpcNSigmaHe, float); //!
DECLARE_SOA_COLUMN(TPCNSigmaAl, tpcNSigmaAl, float); //!
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
DECLARE_SOA_COLUMN(TPCNSigmaStoreEl, tpcNSigmaStoreEl, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreMu, tpcNSigmaStoreMu, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStorePi, tpcNSigmaStorePi, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreKa, tpcNSigmaStoreKa, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStorePr, tpcNSigmaStorePr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreDe, tpcNSigmaStoreDe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreTr, tpcNSigmaStoreTr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreHe, tpcNSigmaStoreHe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(TPCNSigmaStoreAl, tpcNSigmaStoreAl, binned_nsigma_t); //!
// NSigma with reduced size in [binned_min, binned_max] bin size bin_width
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaEl, tpcNSigmaEl); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaMu, tpcNSigmaMu); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaPi, tpcNSigmaPi); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaKa, tpcNSigmaKa); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaPr, tpcNSigmaPr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaDe, tpcNSigmaDe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaTr, tpcNSigmaTr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaHe, tpcNSigmaHe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(TPCNSigmaAl, tpcNSigmaAl); //!

} // namespace pidtpc_tiny

// Per particle tables
DECLARE_SOA_TABLE(pidTPCFullEl, "AOD", "pidTPCFullEl", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for electron
                  pidtpc::TPCExpSignalDiffEl<pidtpc::TPCNSigmaEl, pidtpc::TPCExpSigmaEl>, pidtpc::TPCExpSigmaEl, pidtpc::TPCNSigmaEl);
DECLARE_SOA_TABLE(pidTPCFullMu, "AOD", "pidTPCFullMu", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for muon
                  pidtpc::TPCExpSignalDiffMu<pidtpc::TPCNSigmaMu, pidtpc::TPCExpSigmaMu>, pidtpc::TPCExpSigmaMu, pidtpc::TPCNSigmaMu);
DECLARE_SOA_TABLE(pidTPCFullPi, "AOD", "pidTPCFullPi", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for pion
                  pidtpc::TPCExpSignalDiffPi<pidtpc::TPCNSigmaPi, pidtpc::TPCExpSigmaPi>, pidtpc::TPCExpSigmaPi, pidtpc::TPCNSigmaPi);
DECLARE_SOA_TABLE(pidTPCFullKa, "AOD", "pidTPCFullKa", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for kaon
                  pidtpc::TPCExpSignalDiffKa<pidtpc::TPCNSigmaKa, pidtpc::TPCExpSigmaKa>, pidtpc::TPCExpSigmaKa, pidtpc::TPCNSigmaKa);
DECLARE_SOA_TABLE(pidTPCFullPr, "AOD", "pidTPCFullPr", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for proton
                  pidtpc::TPCExpSignalDiffPr<pidtpc::TPCNSigmaPr, pidtpc::TPCExpSigmaPr>, pidtpc::TPCExpSigmaPr, pidtpc::TPCNSigmaPr);
DECLARE_SOA_TABLE(pidTPCFullDe, "AOD", "pidTPCFullDe", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for deuteron
                  pidtpc::TPCExpSignalDiffDe<pidtpc::TPCNSigmaDe, pidtpc::TPCExpSigmaDe>, pidtpc::TPCExpSigmaDe, pidtpc::TPCNSigmaDe);
DECLARE_SOA_TABLE(pidTPCFullTr, "AOD", "pidTPCFullTr", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for triton
                  pidtpc::TPCExpSignalDiffTr<pidtpc::TPCNSigmaTr, pidtpc::TPCExpSigmaTr>, pidtpc::TPCExpSigmaTr, pidtpc::TPCNSigmaTr);
DECLARE_SOA_TABLE(pidTPCFullHe, "AOD", "pidTPCFullHe", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for helium3
                  pidtpc::TPCExpSignalDiffHe<pidtpc::TPCNSigmaHe, pidtpc::TPCExpSigmaHe>, pidtpc::TPCExpSigmaHe, pidtpc::TPCNSigmaHe);
DECLARE_SOA_TABLE(pidTPCFullAl, "AOD", "pidTPCFullAl", //! Table of the TPC (full) response with expected signal, expected resolution and Nsigma for alpha
                  pidtpc::TPCExpSignalDiffAl<pidtpc::TPCNSigmaAl, pidtpc::TPCExpSigmaAl>, pidtpc::TPCExpSigmaAl, pidtpc::TPCNSigmaAl);

// Tiny size tables
DECLARE_SOA_TABLE(pidTPCEl, "AOD", "pidTPCEl", //! Table of the TPC response with binned Nsigma for electron
                  pidtpc_tiny::TPCNSigmaStoreEl, pidtpc_tiny::TPCNSigmaEl<pidtpc_tiny::TPCNSigmaStoreEl>);
DECLARE_SOA_TABLE(pidTPCMu, "AOD", "pidTPCMu", //! Table of the TPC response with binned Nsigma for muon
                  pidtpc_tiny::TPCNSigmaStoreMu, pidtpc_tiny::TPCNSigmaMu<pidtpc_tiny::TPCNSigmaStoreMu>);
DECLARE_SOA_TABLE(pidTPCPi, "AOD", "pidTPCPi", //! Table of the TPC response with binned Nsigma for pion
                  pidtpc_tiny::TPCNSigmaStorePi, pidtpc_tiny::TPCNSigmaPi<pidtpc_tiny::TPCNSigmaStorePi>);
DECLARE_SOA_TABLE(pidTPCKa, "AOD", "pidTPCKa", //! Table of the TPC response with binned Nsigma for kaon
                  pidtpc_tiny::TPCNSigmaStoreKa, pidtpc_tiny::TPCNSigmaKa<pidtpc_tiny::TPCNSigmaStoreKa>);
DECLARE_SOA_TABLE(pidTPCPr, "AOD", "pidTPCPr", //! Table of the TPC response with binned Nsigma for proton
                  pidtpc_tiny::TPCNSigmaStorePr, pidtpc_tiny::TPCNSigmaPr<pidtpc_tiny::TPCNSigmaStorePr>);
DECLARE_SOA_TABLE(pidTPCDe, "AOD", "pidTPCDe", //! Table of the TPC response with binned Nsigma for deuteron
                  pidtpc_tiny::TPCNSigmaStoreDe, pidtpc_tiny::TPCNSigmaDe<pidtpc_tiny::TPCNSigmaStoreDe>);
DECLARE_SOA_TABLE(pidTPCTr, "AOD", "pidTPCTr", //! Table of the TPC response with binned Nsigma for triton
                  pidtpc_tiny::TPCNSigmaStoreTr, pidtpc_tiny::TPCNSigmaTr<pidtpc_tiny::TPCNSigmaStoreTr>);
DECLARE_SOA_TABLE(pidTPCHe, "AOD", "pidTPCHe", //! Table of the TPC response with binned Nsigma for helium3
                  pidtpc_tiny::TPCNSigmaStoreHe, pidtpc_tiny::TPCNSigmaHe<pidtpc_tiny::TPCNSigmaStoreHe>);
DECLARE_SOA_TABLE(pidTPCAl, "AOD", "pidTPCAl", //! Table of the TPC response with binned Nsigma for alpha
                  pidtpc_tiny::TPCNSigmaStoreAl, pidtpc_tiny::TPCNSigmaAl<pidtpc_tiny::TPCNSigmaStoreAl>);

#undef DEFINE_UNWRAP_NSIGMA_COLUMN

} // namespace o2::aod

#endif // O2_FRAMEWORK_PIDRESPONSE_H_
