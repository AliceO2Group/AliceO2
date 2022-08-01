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

// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright
// holders. All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file ClusterFinderOriginal.cxx
/// \brief Definition of a class to reconstruct clusters with the original MLEM
/// algorithm
///
/// The original code is in AliMUONClusterFinderMLEM and associated classes.
/// It has been re-written in an attempt to simplify it without changing the
/// results.
///
/// \author Gilles Grasseau, Subatech

#ifndef O2_MCH_POISSONEM_H_
#define O2_MCH_POISSONEM_H_

namespace o2
{
namespace mch
{

// namespace  PEM {
// public :
constexpr int nMacroIterations = 8;
static constexpr int nIterations[nMacroIterations] = {5, 10, 10, 10,
                                                      10, 10, 10, 30};
static constexpr double minPadResidues[nMacroIterations] = {2.0, 2.0, 1.5, 1.5,
                                                            1.0, 1.0, 0.5, 0.5};

std::pair<double, double> PoissonEMLoop(const Pads& pads, Pads& pixels,
                                        const double* Cij, Mask_t* maskCij,
                                        int qCutMode, double minPadResidu,
                                        int nItMax, int n0);
// static double computeChiSquare( const Pads &pads, const double
// *qPredictedPads);
//};

} // namespace mch
} // namespace o2

#endif // O2_MCH_POISSONEM_H_
