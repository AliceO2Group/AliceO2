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

/// \file SignalInduction.h
/// \brief Definition of the class handling the signal induction
/// \author Andi Mathis, TU München, andreas.mathis@ph.tum.de

#ifndef ALICEO2_TPC_SignalInduction_H_
#define ALICEO2_TPC_SignalInduction_H_

#include "TPCBase/Mapper.h"
#include "TPCBase/ParameterGEM.h"
#include "DataFormatsTPC/Defs.h"
#include "TGraph2D.h"

namespace o2
{
namespace tpc
{

/// \class SignalInduction
/// This class takes care of the signal induction on the pad plane.
/// The actual Pad Response Function (PRF) is simulated with Garfield++/COMSOL and dumped to a file.
/// This file is read by this class and buffered for each pad size individually (IROC / OROC1-2 / OROC3).

struct PadResponse {
  PadResponse() = default;
  PadResponse(const DigitPos& digiPos, double padResp)
    : digiPos{digiPos}, padResp{padResp}
  {
  }
  DigitPos digiPos;    ///< DigitPos of the individual pad
  double padResp = -1; ///< Corresponding weight of the electron signal
};

class SignalInduction
{
 public:
  static SignalInduction& instance()
  {
    static SignalInduction signalInduction;
    return signalInduction;
  }

  /// Destructor
  ~SignalInduction() = default;

  /// Compute the pad response on a given pad for a given electron arrival position
  /// \param offsetX Offset in X w.r.t. the pad center
  /// \param offsetY Offset in Y w.r.t. the pad center
  /// \param gemstack GEM stack where the electron arrives
  /// \return Weight for the contribution to the induced signal
  double computePadResponse(const double offsetX, const double offsetY, const GEMstack gemstack) const;

  /// Compute the pad response on a given pad for a given electron arrival position
  /// \param posEle Arrival position of the electron
  /// \param digitPos DigitPos to consider
  /// \param offsetPad Pad offset w.r.t. electron arrival position
  /// \param offsetRow Row offset w.r.t. electron arrival position
  /// \return Weight for the contribution to the induced signal on that pad
  PadResponse computePadResponse(const GlobalPosition3D posEle, const DigitPos digitPos, const double offsetPad, const double offsetRow) const;

  /// Compute the pad response on a given pad for a given electron arrival position
  /// \param posEle Arrival position of the electron
  /// \param digitPos DigitPos to consider
  /// \param signalArray Array filled with the pad response values
  void getPadResponse(const GlobalPosition3D posEle, const DigitPos digitPos, std::vector<PadResponse>& signalArray, const PadResponseMode& mode) const;

 private:
  SignalInduction();

  /// Load and buffer the pad response function from the data files
  void loadPadResponse();

  TGraph2D* grPRFIROC;
  TGraph2D* grPRFOROC12;
  TGraph2D* grPRFOROC3;
};

inline double SignalInduction::computePadResponse(const double offsetX, const double offsetY, const GEMstack gemstack) const
{
  // TODO The TGraph2D::Interpolate method is certainly performance-wise far from being ideal - to be replaced
  switch (gemstack) {
    case IROCgem: {
      return grPRFIROC->Interpolate(offsetY, offsetX);
      break;
    }
    case OROC1gem:
    case OROC2gem: {
      return grPRFOROC12->Interpolate(offsetY, offsetX);
      break;
    }
    case OROC3gem: {
      return grPRFOROC3->Interpolate(offsetY, offsetX);
      break;
    }
  }
  return -1.;
}

} // namespace tpc
} // namespace o2

#endif // ALICEO2_TPC_SignalInduction_H_
