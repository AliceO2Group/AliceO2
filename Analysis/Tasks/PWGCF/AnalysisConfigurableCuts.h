// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.
#ifndef ANALYSIS_CONFIGURABLE_CUTS_CLASSES_H
#define ANALYSIS_CONFIGURABLE_CUTS_CLASSES_H

#include <Rtypes.h>
#include <TObject.h>
#include <TNamed.h>
#include <TMath.h>

namespace o2
{
namespace analysis
{
/// \class EventSelectionCuts
/// \brief Class which implements configurable event selection cuts
///
class EventSelectionCuts
{
 public:
  int mOfflinetrigger = 1;                     ///< offline trigger, default MB = 1
  std::string mCentmultestmator = "V0M";       ///< centrality / multiplicity estimation, default V0M
  int mRemovepileupcode = 1;                   ///< Procedure for pile-up removal, default V0M vs TPCout tracks = 1
  std::string mRemovepileupfn = "-2500+5.0*x"; ///< function for pile-up removal, procedure dependent, defaul V0M vs TPCout tracks for LHC15o HIR

 private:
  ClassDefNV(EventSelectionCuts, 1);
};

/// \class DptDptBinningCuts
/// \brief Class which implements configurable acceptance cuts
///
class DptDptBinningCuts
{
 public:
  int mZVtxbins = 28;             ///< the number of z_vtx bins default 28
  float mZVtxmin = -7.0;          ///< the minimum z_vtx value, default -7.0 cm
  float mZVtxmax = 7.0;           ///< the maximum z_vtx value, default 7.0 cm
  int mPTbins = 18;               ///< the number of pT bins, default 18
  float mPTmin = 0.2;             ///< the minimum pT value, default 0.2 GeV
  float mPTmax = 2.0;             ///< the maximum pT value, default 2.0 GeV
  int mEtabins = 16;              ///< the number of eta bins default 16
  float mEtamin = -0.8;           ///< the minimum eta value, default -0.8
  float mEtamax = 0.8;            ///< the maximum eta value, default 0.8
  int mPhibins = 72;              ///< the number of phi bins, default 72
  float mPhimin = 0.0;            ///< the minimum phi value, default 0.0
  float mPhimax = TMath::TwoPi(); ///< the maximum phi value, default 2*pi
  float mPhibinshift = 0.5;       ///< the shift in the azimuthal origen, defoult 0.5, i.e half a bin

 private:
  ClassDefNV(DptDptBinningCuts, 1);
};

class SimpleInclusiveCut : public TNamed
{
 public:
  int mX = 1;
  float mY = 2.f;
  SimpleInclusiveCut();
  SimpleInclusiveCut(const char*, int, float);
  SimpleInclusiveCut(const SimpleInclusiveCut&) = default;
  ~SimpleInclusiveCut() override = default;

  SimpleInclusiveCut& operator=(const SimpleInclusiveCut&);

 private:
  ClassDef(SimpleInclusiveCut, 1);
};

} // namespace analysis
} // namespace o2
#endif // ANALYSIS_CONFIGURABLE_CUTS_CLASSES_H
