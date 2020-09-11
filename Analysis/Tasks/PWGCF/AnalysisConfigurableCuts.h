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
class Binning
{
 public:
  int nbins = 0;
  float minval = 0.0;
  float maxval = 0.0;
  Binning() : nbins(0), minval(0.0), maxval(0.0){};
  Binning(int _n, float _mi, float _ma) : nbins(_n), minval(_mi), maxval(_ma){};
  bool isAccepted(float value)
  {
    return (minval < value and value < maxval);
  }

 private:
  ClassDefNV(Binning, 1);
};

class DptDptBinning
{
 public:
  Binning pTBinning = {18, 0.2, 2.0};
  Binning etaBinnig = {16, -0.8, 0.8};
  Binning phiBinning = {72, 0.0, TMath::TwoPi()};
  Binning zVtxBinning = {28, -7.0, 7.0};
  bool isPtAccepted(float pT)
  {
    return pTBinning.isAccepted(pT);
  }

 private:
  ClassDefNV(DptDptBinning, 1);
};

class SimpleInclusiveCut : public TNamed
{
 public:
  int x = 1;
  float y = 2.f;
  SimpleInclusiveCut();
  SimpleInclusiveCut(const char*, int, float);
  SimpleInclusiveCut(const SimpleInclusiveCut&);
  virtual ~SimpleInclusiveCut();

  SimpleInclusiveCut& operator=(const SimpleInclusiveCut&);

 private:
  ClassDef(SimpleInclusiveCut, 1);
};

} // namespace analysis
} // namespace o2
#endif // ANALYSIS_CONFIGURABLE_CUTS_CLASSES_H
