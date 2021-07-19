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

///
/// \file   OuterTOF.h
/// \author Nicolo' Jacazio
/// \since  19/07/2021
/// \brief  Set of tables for the ALICE3 outer barrel TOF information
///

#ifndef O2_ANALYSIS_ALICE3_OBTOF_H_
#define O2_ANALYSIS_ALICE3_OBTOF_H_

// O2 includes
#include "Framework/ASoA.h"
#include "Framework/AnalysisDataModel.h"

namespace o2::aod
{
namespace alice3obtof
{
DECLARE_SOA_COLUMN(OBTOFSignal, obtofSignal, float); //! signal of the outer barrel TOF matched to the track
DECLARE_SOA_COLUMN(OBLength, oblength, float);       //! Track length matched to the outer barrel TOF
DECLARE_SOA_COLUMN(OBTOFExpMom, obtofExpMom, float); //! TOF expected momentum obtained in tracking, used to compute the expected times
DECLARE_SOA_DYNAMIC_COLUMN(HasOBTOF, hasOBTOF,       //! Flag to check if track has a TOF measurement
                           [](float tofSignal, float tofExpMom) -> bool { return (tofSignal > 0.f) && (tofExpMom > 0.f); });
} // namespace alice3obtof

DECLARE_SOA_TABLE(OBTOFs, "AOD", "OBTOF", //! Additional track information for the ALICE3 Outer Barrel TOF
                  alice3obtof::OBTOFSignal, alice3obtof::OBLength, alice3obtof::OBTOFExpMom,
                  alice3obtof::HasOBTOF<alice3obtof::OBTOFSignal, alice3obtof::OBTOFExpMom>);

using OBTOF = OBTOFs::iterator;

// NSigma values
namespace pidobtof
{
// Expected times
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffEl, obtofExpSignalDiffEl, //! Difference between signal and expected for electron
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffMu, obtofExpSignalDiffMu, //! Difference between signal and expected for muon
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffPi, obtofExpSignalDiffPi, //! Difference between signal and expected for pion
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffKa, obtofExpSignalDiffKa, //! Difference between signal and expected for kaon
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffPr, obtofExpSignalDiffPr, //! Difference between signal and expected for proton
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffDe, obtofExpSignalDiffDe, //! Difference between signal and expected for deuteron
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffTr, obtofExpSignalDiffTr, //! Difference between signal and expected for triton
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffHe, obtofExpSignalDiffHe, //! Difference between signal and expected for helium3
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
DECLARE_SOA_DYNAMIC_COLUMN(OBTOFExpSignalDiffAl, obtofExpSignalDiffAl, //! Difference between signal and expected for alpha
                           [](float nsigma, float sigma) -> float { return nsigma * sigma; });
// Expected sigma
DECLARE_SOA_COLUMN(OBTOFExpSigmaEl, obtofExpSigmaEl, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaMu, obtofExpSigmaMu, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaPi, obtofExpSigmaPi, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaKa, obtofExpSigmaKa, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaPr, obtofExpSigmaPr, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaDe, obtofExpSigmaDe, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaTr, obtofExpSigmaTr, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaHe, obtofExpSigmaHe, float); //!
DECLARE_SOA_COLUMN(OBTOFExpSigmaAl, obtofExpSigmaAl, float); //!
// NSigma
DECLARE_SOA_COLUMN(OBTOFNSigmaEl, obtofNSigmaEl, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaMu, obtofNSigmaMu, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaPi, obtofNSigmaPi, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaKa, obtofNSigmaKa, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaPr, obtofNSigmaPr, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaDe, obtofNSigmaDe, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaTr, obtofNSigmaTr, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaHe, obtofNSigmaHe, float); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaAl, obtofNSigmaAl, float); //!
} // namespace pidobtof

// Macro to convert the stored Nsigmas to floats
#define DEFINE_UNWRAP_NSIGMA_COLUMN(COLUMN, COLUMN_NAME) \
  DECLARE_SOA_DYNAMIC_COLUMN(COLUMN, COLUMN_NAME,        \
                             [](binned_nsigma_t nsigma_binned) -> float { return bin_width * static_cast<float>(nsigma_binned); });

namespace pidobtof_tiny
{
typedef int8_t binned_nsigma_t;
constexpr int nbins = (1 << 8 * sizeof(binned_nsigma_t)) - 2;
constexpr binned_nsigma_t upper_bin = nbins >> 1;
constexpr binned_nsigma_t lower_bin = -(nbins >> 1);
constexpr float binned_max = 6.35;
constexpr float binned_min = -6.35;
constexpr float bin_width = (binned_max - binned_min) / nbins;
// NSigma with reduced size 8 bit
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreEl, obtofNSigmaStoreEl, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreMu, obtofNSigmaStoreMu, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStorePi, obtofNSigmaStorePi, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreKa, obtofNSigmaStoreKa, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStorePr, obtofNSigmaStorePr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreDe, obtofNSigmaStoreDe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreTr, obtofNSigmaStoreTr, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreHe, obtofNSigmaStoreHe, binned_nsigma_t); //!
DECLARE_SOA_COLUMN(OBTOFNSigmaStoreAl, obtofNSigmaStoreAl, binned_nsigma_t); //!
// NSigma with reduced size in [binned_min, binned_max] bin size bin_width
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaEl, obtofNSigmaEl); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaMu, obtofNSigmaMu); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaPi, obtofNSigmaPi); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaKa, obtofNSigmaKa); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaPr, obtofNSigmaPr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaDe, obtofNSigmaDe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaTr, obtofNSigmaTr); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaHe, obtofNSigmaHe); //!
DEFINE_UNWRAP_NSIGMA_COLUMN(OBTOFNSigmaAl, obtofNSigmaAl); //!

} // namespace pidobtof_tiny

// Per particle tables
DECLARE_SOA_TABLE(pidOBTOFFullEl, "AOD", "pidOBTOFFullEl", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for electron
                  pidobtof::OBTOFExpSignalDiffEl<pidobtof::OBTOFNSigmaEl, pidobtof::OBTOFExpSigmaEl>, pidobtof::OBTOFExpSigmaEl, pidobtof::OBTOFNSigmaEl);
DECLARE_SOA_TABLE(pidOBTOFFullMu, "AOD", "pidOBTOFFullMu", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for muon
                  pidobtof::OBTOFExpSignalDiffMu<pidobtof::OBTOFNSigmaMu, pidobtof::OBTOFExpSigmaMu>, pidobtof::OBTOFExpSigmaMu, pidobtof::OBTOFNSigmaMu);
DECLARE_SOA_TABLE(pidOBTOFFullPi, "AOD", "pidOBTOFFullPi", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for pion
                  pidobtof::OBTOFExpSignalDiffPi<pidobtof::OBTOFNSigmaPi, pidobtof::OBTOFExpSigmaPi>, pidobtof::OBTOFExpSigmaPi, pidobtof::OBTOFNSigmaPi);
DECLARE_SOA_TABLE(pidOBTOFFullKa, "AOD", "pidOBTOFFullKa", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for kaon
                  pidobtof::OBTOFExpSignalDiffKa<pidobtof::OBTOFNSigmaKa, pidobtof::OBTOFExpSigmaKa>, pidobtof::OBTOFExpSigmaKa, pidobtof::OBTOFNSigmaKa);
DECLARE_SOA_TABLE(pidOBTOFFullPr, "AOD", "pidOBTOFFullPr", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for proton
                  pidobtof::OBTOFExpSignalDiffPr<pidobtof::OBTOFNSigmaPr, pidobtof::OBTOFExpSigmaPr>, pidobtof::OBTOFExpSigmaPr, pidobtof::OBTOFNSigmaPr);
DECLARE_SOA_TABLE(pidOBTOFFullDe, "AOD", "pidOBTOFFullDe", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for deuteron
                  pidobtof::OBTOFExpSignalDiffDe<pidobtof::OBTOFNSigmaDe, pidobtof::OBTOFExpSigmaDe>, pidobtof::OBTOFExpSigmaDe, pidobtof::OBTOFNSigmaDe);
DECLARE_SOA_TABLE(pidOBTOFFullTr, "AOD", "pidOBTOFFullTr", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for triton
                  pidobtof::OBTOFExpSignalDiffTr<pidobtof::OBTOFNSigmaTr, pidobtof::OBTOFExpSigmaTr>, pidobtof::OBTOFExpSigmaTr, pidobtof::OBTOFNSigmaTr);
DECLARE_SOA_TABLE(pidOBTOFFullHe, "AOD", "pidOBTOFFullHe", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for helium3
                  pidobtof::OBTOFExpSignalDiffHe<pidobtof::OBTOFNSigmaHe, pidobtof::OBTOFExpSigmaHe>, pidobtof::OBTOFExpSigmaHe, pidobtof::OBTOFNSigmaHe);
DECLARE_SOA_TABLE(pidOBTOFFullAl, "AOD", "pidOBTOFFullAl", //! Table of the OBTOF (full) response with expected signal, expected resolution and Nsigma for alpha
                  pidobtof::OBTOFExpSignalDiffAl<pidobtof::OBTOFNSigmaAl, pidobtof::OBTOFExpSigmaAl>, pidobtof::OBTOFExpSigmaAl, pidobtof::OBTOFNSigmaAl);

// // Tiny size tables
DECLARE_SOA_TABLE(pidOBTOFEl, "AOD", "pidOBTOFEl", //! Table of the OBTOF response with binned Nsigma for electron
                  pidobtof_tiny::OBTOFNSigmaStoreEl, pidobtof_tiny::OBTOFNSigmaEl<pidobtof_tiny::OBTOFNSigmaStoreEl>);
DECLARE_SOA_TABLE(pidOBTOFMu, "AOD", "pidOBTOFMu", //! Table of the OBTOF response with binned Nsigma for muon
                  pidobtof_tiny::OBTOFNSigmaStoreMu, pidobtof_tiny::OBTOFNSigmaMu<pidobtof_tiny::OBTOFNSigmaStoreMu>);
DECLARE_SOA_TABLE(pidOBTOFPi, "AOD", "pidOBTOFPi", //! Table of the OBTOF response with binned Nsigma for pion
                  pidobtof_tiny::OBTOFNSigmaStorePi, pidobtof_tiny::OBTOFNSigmaPi<pidobtof_tiny::OBTOFNSigmaStorePi>);
DECLARE_SOA_TABLE(pidOBTOFKa, "AOD", "pidOBTOFKa", //! Table of the OBTOF response with binned Nsigma for kaon
                  pidobtof_tiny::OBTOFNSigmaStoreKa, pidobtof_tiny::OBTOFNSigmaKa<pidobtof_tiny::OBTOFNSigmaStoreKa>);
DECLARE_SOA_TABLE(pidOBTOFPr, "AOD", "pidOBTOFPr", //! Table of the OBTOF response with binned Nsigma for proton
                  pidobtof_tiny::OBTOFNSigmaStorePr, pidobtof_tiny::OBTOFNSigmaPr<pidobtof_tiny::OBTOFNSigmaStorePr>);
DECLARE_SOA_TABLE(pidOBTOFDe, "AOD", "pidOBTOFDe", //! Table of the OBTOF response with binned Nsigma for deuteron
                  pidobtof_tiny::OBTOFNSigmaStoreDe, pidobtof_tiny::OBTOFNSigmaDe<pidobtof_tiny::OBTOFNSigmaStoreDe>);
DECLARE_SOA_TABLE(pidOBTOFTr, "AOD", "pidOBTOFTr", //! Table of the OBTOF response with binned Nsigma for triton
                  pidobtof_tiny::OBTOFNSigmaStoreTr, pidobtof_tiny::OBTOFNSigmaTr<pidobtof_tiny::OBTOFNSigmaStoreTr>);
DECLARE_SOA_TABLE(pidOBTOFHe, "AOD", "pidOBTOFHe", //! Table of the OBTOF response with binned Nsigma for helium3
                  pidobtof_tiny::OBTOFNSigmaStoreHe, pidobtof_tiny::OBTOFNSigmaHe<pidobtof_tiny::OBTOFNSigmaStoreHe>);
DECLARE_SOA_TABLE(pidOBTOFAl, "AOD", "pidOBTOFAl", //! Table of the OBTOF response with binned Nsigma for alpha
                  pidobtof_tiny::OBTOFNSigmaStoreAl, pidobtof_tiny::OBTOFNSigmaAl<pidobtof_tiny::OBTOFNSigmaStoreAl>);

#undef DEFINE_UNWRAP_NSIGMA_COLUMN

} // namespace o2::aod

#endif // O2_ANALYSIS_ALICE3_OBTOF_H_
