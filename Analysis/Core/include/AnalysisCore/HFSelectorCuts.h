// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file HFSelectorCuts.h
/// \brief Default pT bins and cut arrays for heavy-flavour selectors and analysis tasks

#ifndef HF_SELECTOR_CUTS_H_
#define HF_SELECTOR_CUTS_H_

#include "Framework/Configurable.h"
#include <vector>
#include <string>

namespace o2::analysis
{
namespace pdg
{
enum Code {
  kD0 = 421,
  kD0bar = -421,
  kDPlus = 411,
  kLambdaCPlus = 4122,
  kXiCPlus = 4232,
  kJpsi = 443
};
} // namespace pdg

/// Finds pT bin in a configurable array.
/// \param bins  array of pT bins
/// \param value  pT
/// \return index of the pT bin
/// \note Accounts for the offset so that pt bin array can be used to also configure a histogram axis.
template <typename T1, typename T2>
int findBin(o2::framework::Configurable<std::vector<T1>> const& bins, T2 value)
{
  if (value < bins->front()) {
    return -1;
  }
  if (value >= bins->back()) {
    return -1;
  }
  return std::distance(bins->begin(), std::upper_bound(bins->begin(), bins->end(), value)) - 1;
}

// namespace per channel

namespace hf_cuts_single_track
{
static constexpr int npTBinsTrack = 6;
static constexpr int nCutVarsTrack = 2;
// default values for the pT bin edges (can be used to configure histogram axis)
// common for any candidate type (2-prong, 3-prong)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBinsTrack[npTBinsTrack + 1] = {
  0,
  0.5,
  1.0,
  1.5,
  2.0,
  3.0,
  1000.0};
auto pTBinsTrack_v = std::vector<double>{pTBinsTrack, pTBinsTrack + npTBinsTrack + 1};

// default values for the cuts
constexpr double cutsTrack[npTBinsTrack][nCutVarsTrack] = {{0., 10.},  /* pt<0.5*/
                                                           {0., 10.},  /* 0.5<pt<1*/
                                                           {0., 10.},  /* 1<pt<1.5*/
                                                           {0., 10.},  /* 1.5<pt<2*/
                                                           {0., 10.},  /* 2<pt<3*/
                                                           {0., 10.}}; /* pt>3*/

// row labels
static const std::vector<std::string> pTBinLabelsTrack{};

// column labels
static const std::vector<std::string> cutVarLabelsTrack = {"min_dcaxytoprimary", "max_dcaxytoprimary"};
} // namespace hf_cuts_single_track

namespace hf_cuts_d0_topik
{
static constexpr int npTBins = 25;
static constexpr int nCutVars = 11;
// default values for the pT bin edges (can be used to configure histogram axis)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBins[npTBins + 1] = {
  0,
  0.5,
  1.0,
  1.5,
  2.0,
  2.5,
  3.0,
  3.5,
  4.0,
  4.5,
  5.0,
  5.5,
  6.0,
  6.5,
  7.0,
  7.5,
  8.0,
  9.0,
  10.0,
  12.0,
  16.0,
  20.0,
  24.0,
  36.0,
  50.0,
  100.0};
auto pTBins_v = std::vector<double>{pTBins, pTBins + npTBins + 1};

// default values for the cuts
constexpr double cuts[npTBins][nCutVars] = {{0.400, 350. * 1E-4, 0.8, 0.5, 0.5, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.80, 0., 0.},   /* 0   < pT < 0.5 */
                                            {0.400, 350. * 1E-4, 0.8, 0.5, 0.5, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.80, 0., 0.},   /* 0.5 < pT < 1   */
                                            {0.400, 300. * 1E-4, 0.8, 0.4, 0.4, 1000. * 1E-4, 1000. * 1E-4, -25000. * 1E-8, 0.80, 0., 0.},  /* 1   < pT < 1.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.4, 0.4, 1000. * 1E-4, 1000. * 1E-4, -25000. * 1E-8, 0.80, 0., 0.},  /* 1.5 < pT < 2   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -20000. * 1E-8, 0.90, 0., 0.},  /* 2   < pT < 2.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -20000. * 1E-8, 0.90, 0., 0.},  /* 2.5 < pT < 3   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -12000. * 1E-8, 0.85, 0., 0.},  /* 3   < pT < 3.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -12000. * 1E-8, 0.85, 0., 0.},  /* 3.5 < pT < 4   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 4   < pT < 4.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 4.5 < pT < 5   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 5   < pT < 5.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 5.5 < pT < 6   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 6   < pT < 6.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -8000. * 1E-8, 0.85, 0., 0.},   /* 6.5 < pT < 7   */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -7000. * 1E-8, 0.85, 0., 0.},   /* 7   < pT < 7.5 */
                                            {0.400, 300. * 1E-4, 0.8, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -7000. * 1E-8, 0.85, 0., 0.},   /* 7.5 < pT < 8   */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 8   < pT < 9   */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 9   < pT < 10  */
                                            {0.400, 300. * 1E-4, 0.9, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, -5000. * 1E-8, 0.85, 0., 0.},   /* 10  < pT < 12  */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 10000. * 1E-8, 0.85, 0., 0.},   /* 12  < pT < 16  */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 16  < pT < 20  */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 20  < pT < 24  */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 24  < pT < 36  */
                                            {0.400, 300. * 1E-4, 1.0, 0.7, 0.7, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.85, 0., 0.},  /* 36  < pT < 50  */
                                            {0.400, 300. * 1E-4, 1.0, 0.6, 0.6, 1000. * 1E-4, 1000. * 1E-4, 999999. * 1E-8, 0.80, 0., 0.}}; /* 50  < pT < 100 */

// row labels
static const std::vector<std::string> pTBinLabels = {
  "pT bin 0",
  "pT bin 1",
  "pT bin 2",
  "pT bin 3",
  "pT bin 4",
  "pT bin 5",
  "pT bin 6",
  "pT bin 7",
  "pT bin 8",
  "pT bin 9",
  "pT bin 10",
  "pT bin 11",
  "pT bin 12",
  "pT bin 13",
  "pT bin 14",
  "pT bin 15",
  "pT bin 16",
  "pT bin 17",
  "pT bin 18",
  "pT bin 19",
  "pT bin 20",
  "pT bin 21",
  "pT bin 22",
  "pT bin 23",
  "pT bin 24"};

// column labels
static const std::vector<std::string> cutVarLabels = {"m", "DCA", "cos theta*", "pT K", "pT Pi", "d0K", "d0pi", "d0d0", "cos pointing angle", "cos pointing angle xy", "normalized decay length XY"};
} // namespace hf_cuts_d0_topik

namespace hf_cuts_lc_topkpi
{
static constexpr int npTBins = 10;
static constexpr int nCutVars = 7;
// default values for the pT bin edges (can be used to configure histogram axis)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBins[npTBins + 1] = {
  0.,
  1.,
  2.,
  3.,
  4.,
  5.,
  6.,
  8.,
  12.,
  24.,
  36.};
auto pTBins_v = std::vector<double>{pTBins, pTBins + npTBins + 1};

// default values for the cuts
constexpr double cuts[npTBins][nCutVars] = {{0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 0  < pT < 1  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 1  < pT < 2  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 2  < pT < 3  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 3  < pT < 4  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 4  < pT < 5  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 5  < pT < 6  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 6  < pT < 8  */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 8  < pT < 12 */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.},  /* 12 < pT < 24 */
                                            {0.400, 0.4, 0.4, 0.4, 0., 0.005, 0.}}; /* 24 < pT < 36 */

// row labels
static const std::vector<std::string> pTBinLabels = {
  "pT bin 0",
  "pT bin 1",
  "pT bin 2",
  "pT bin 3",
  "pT bin 4",
  "pT bin 5",
  "pT bin 6",
  "pT bin 7",
  "pT bin 8",
  "pT bin 9"};

// column labels
static const std::vector<std::string> cutVarLabels = {"m", "pT p", "pT K", "pT Pi", "Chi2PCA", "decay length", "cos pointing angle"};
} // namespace hf_cuts_lc_topkpi

namespace hf_cuts_dplus_topikpi
{
static const int npTBins = 12;
static const int nCutVars = 8;
// default values for the pT bin edges (can be used to configure histogram axis)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBins[npTBins + 1] = {
  1.,
  2.,
  3.,
  4.,
  5.,
  6.,
  7.,
  8.,
  10.,
  12.,
  16.,
  24.,
  36.};
auto pTBins_v = std::vector<double>{pTBins, pTBins + npTBins + 1};

// default values for the cuts
// selections from pp at 5 TeV 2017 analysis https://alice-notes.web.cern.ch/node/808
constexpr double cuts[npTBins][nCutVars] = {{0.2, 0.3, 0.3, 0.07, 6., 0.96, 0.985, 2.5},  /* 1  < pT < 2  */
                                            {0.2, 0.3, 0.3, 0.07, 5., 0.96, 0.985, 2.5},  /* 2  < pT < 3  */
                                            {0.2, 0.3, 0.3, 0.10, 5., 0.96, 0.980, 2.5},  /* 3  < pT < 4  */
                                            {0.2, 0.3, 0.3, 0.10, 5., 0.96, 0.000, 2.5},  /* 4  < pT < 5  */
                                            {0.2, 0.3, 0.3, 0.10, 5., 0.96, 0.000, 2.5},  /* 5  < pT < 6  */
                                            {0.2, 0.3, 0.3, 0.10, 5., 0.96, 0.000, 2.5},  /* 6  < pT < 7  */
                                            {0.2, 0.3, 0.3, 0.10, 5., 0.96, 0.000, 2.5},  /* 7  < pT < 8  */
                                            {0.2, 0.3, 0.3, 0.12, 5., 0.96, 0.000, 2.5},  /* 8  < pT < 10 */
                                            {0.2, 0.3, 0.3, 0.12, 5., 0.96, 0.000, 2.5},  /* 10 < pT < 12 */
                                            {0.2, 0.3, 0.3, 0.12, 5., 0.96, 0.000, 2.5},  /* 12 < pT < 16 */
                                            {0.2, 0.3, 0.3, 0.12, 5., 0.96, 0.000, 2.5},  /* 16 < pT < 24 */
                                            {0.2, 0.3, 0.3, 0.20, 5., 0.94, 0.000, 2.5}}; /* 24 < pT < 36 */

// row labels
static const std::vector<std::string> pTBinLabels = {
  "pT bin 0",
  "pT bin 1",
  "pT bin 2",
  "pT bin 3",
  "pT bin 4",
  "pT bin 5",
  "pT bin 6",
  "pT bin 7",
  "pT bin 8",
  "pT bin 9",
  "pT bin 10",
  "pT bin 11"};

// column labels
static const std::vector<std::string> cutVarLabels = {"deltaM", "pT Pi", "pT K", "decay length", "normalized decay length XY", "cos pointing angle", "cos pointing angle XY", "max normalized deltaIP"};
} // namespace hf_cuts_dplus_topikpi

namespace hf_cuts_xic_topkpi
{
static const int npTBins = 10;
static const int nCutVars = 8;
// default values for the pT bin edges (can be used to configure histogram axis)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBins[npTBins + 1] = {
  0.,
  1.,
  2.,
  3.,
  4.,
  5.,
  6.,
  8.,
  12.,
  24.,
  36.};
auto pTBins_v = std::vector<double>{pTBins, pTBins + npTBins + 1};

// default values for the cuts
constexpr double cuts[npTBins][nCutVars] = {{0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 0  < pT < 1  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 1  < pT < 2  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 2  < pT < 3  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 3  < pT < 4  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 4  < pT < 5  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 5  < pT < 6  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 6  < pT < 8  */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 8  < pT < 12 */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.},  /* 12 < pT < 24 */
                                            {0.400, 0.4, 0.4, 0.4, 0.05, 0.09, 0.005, 0.}}; /* 24 < pT < 36 */

// row labels
static const std::vector<std::string> pTBinLabels = {
  "pT bin 0",
  "pT bin 1",
  "pT bin 2",
  "pT bin 3",
  "pT bin 4",
  "pT bin 5",
  "pT bin 6",
  "pT bin 7",
  "pT bin 8",
  "pT bin 9"};

// column labels
static const std::vector<std::string> cutVarLabels = {"m", "pT p", "pT K", "pT Pi", "DCA", "vertex sigma", "decay length", "cos pointing angle"};
} // namespace hf_cuts_xic_topkpi

namespace hf_cuts_jpsi_toee
{
static constexpr int npTBins = 9;
static constexpr int nCutVars = 4;
// default values for the pT bin edges (can be used to configure histogram axis)
// offset by 1 from the bin numbers in cuts array
constexpr double pTBins[npTBins + 1] = {
  0,
  0.5,
  1.0,
  2.0,
  3.0,
  4.0,
  5.0,
  7.0,
  10.0,
  15.0,
};
auto pTBins_v = std::vector<double>{pTBins, pTBins + npTBins + 1};

// default values for the cuts
constexpr double cuts[npTBins][nCutVars] = {{0.5, 0.2, 0.4, 1},  /* 0   < pT < 0.5 */
                                            {0.5, 0.2, 0.4, 1},  /* 0.5 < pT < 1   */
                                            {0.5, 0.2, 0.4, 1},  /* 1   < pT < 2   */
                                            {0.5, 0.2, 0.4, 1},  /* 2   < pT < 3   */
                                            {0.5, 0.2, 0.4, 1},  /* 3   < pT < 4   */
                                            {0.5, 0.2, 0.4, 1},  /* 4   < pT < 5   */
                                            {0.5, 0.2, 0.4, 1},  /* 5   < pT < 7   */
                                            {0.5, 0.2, 0.4, 1},  /* 7   < pT < 10  */
                                            {0.5, 0.2, 0.4, 1}}; /* 10  < pT < 15  */

// row labels
static const std::vector<std::string> pTBinLabels = {
  "pT bin 0",
  "pT bin 1",
  "pT bin 2",
  "pT bin 3",
  "pT bin 4",
  "pT bin 5",
  "pT bin 6",
  "pT bin 7",
  "pT bin 8"};

// column labels
static const std::vector<std::string> cutVarLabels = {"m", "DCA_xy", "DCA_z", "pT El"};
} // namespace hf_cuts_jpsi_toee
} // namespace o2::analysis

#endif // HF_SELECTOR_CUTS_H_
