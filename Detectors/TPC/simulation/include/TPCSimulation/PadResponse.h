// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file PadResponse.h
/// \brief Definition of the Pad Response
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_PadResponse_H_
#define ALICEO2_TPC_PadResponse_H_

#include "TGraph2D.h"

#include "TPCBase/Mapper.h"

namespace o2
{
namespace tpc
{

/// \class PadResponse
/// This class deals with the induction of the signal on the pad plane.
/// The actual Pad Response Function (PRF) is simulated with Garfield++/COMSOL and dumped to a file.
/// This file is read by this class and dumped to a TGraph2D for each pad size individually (IROC / OROC1-2 / OROC3).
/// The pad response is then computed by evaluating this TGraph2D for a given pad size by evaluating the PRF at the electron position with respect to the pad centre.

class PadResponse
{
 public:
  /// Default constructor
  PadResponse();

  /// Destructor
  virtual ~PadResponse() = default;

  /// Import the PRF from a .dat file to a TGraph2D
  /// \param file Name of the .dat file
  /// \param grPRF TGraph2D to which the PRF will be written
  /// \return Boolean if succesful or not
  bool importPRF(std::string file, std::unique_ptr<TGraph2D>& grPRF) const;

  /// Compute the impact of the pad response for electrons arriving at the GEM stack
  /// \param posEle Position of the electron in real space
  /// \param digiPadPos Position of the electron in pad space
  /// \return Normalized pad response
  float getPadResponse(GlobalPosition3D posEle, DigitPos digiPadPos) const;

 private:
  std::unique_ptr<TGraph2D> mIROC;   ///< TGraph2D holding the PRF for the IROC (4x7.5 mm2 pads)
  std::unique_ptr<TGraph2D> mOROC12; ///< TGraph2D holding the PRF for the OROC1 and OROC2 (6x10 mm2 pads)
  std::unique_ptr<TGraph2D> mOROC3;  ///< TGraph2D holding the PRF for the OROC3 (6x15 mm2 pads)
};
} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_PadResponse_H_
