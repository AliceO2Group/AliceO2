// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/// \file TrackFitterSpec.h
/// \brief Definition of a data processor to read, refit and send tracks with attached clusters
///
/// \author Philippe Pillot, Subatech; adapted by Rafael Pezzi, UFRGS

#ifndef ALICEO2_MFT_TRACKFITTERSPEC_H_
#define ALICEO2_MFT_TRACKFITTERSPEC_H_

#include "MFTTracking/TrackFitter.h"
#include "DataFormatsParameters/GRPObject.h"
#include "Framework/DataProcessorSpec.h"
#include "Framework/Task.h"

namespace o2
{
namespace mft
{

using SMatrix55 = ROOT::Math::SMatrix<double, 5, 5, ROOT::Math::MatRepSym<double, 5>>;
using SMatrix5 = ROOT::Math::SVector<Double_t, 5>;

class TrackFitterTask : public o2::framework::Task
{
 public:
  TrackFitterTask(bool useMC) : mUseMC(useMC) {}
  ~TrackFitterTask() override = default;
  void init(o2::framework::InitContext& ic) final;
  void run(o2::framework::ProcessingContext& pc) final;

 private:
  int mState = 0;
  bool mUseMC = true;
  std::unique_ptr<o2::parameters::GRPObject> mGRP = nullptr;
  std::unique_ptr<o2::mft::TrackFitter> mTrackFitter = nullptr;
};

template <typename T, typename O, typename C>
void convertTrack(const T& inTrack, O& outTrack, C& clusters);

SMatrix55 TtoSMatrixSym55(TMatrixD inMatrix);
SMatrix5 TtoSMatrix5(TMatrixD inMatrix);

o2::framework::DataProcessorSpec getTrackFitterSpec(bool useMC);

} // end namespace mft
} // end namespace o2

#endif // ALICEO2_MFT_TRACKFITTERSPEC_H_
