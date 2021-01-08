// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef CONFIGURABLECUT_H
#define CONFIGURABLECUT_H

#include "Framework/Array2D.h"
#include <iosfwd>
#include <Rtypes.h>
#include <TMath.h>

static constexpr double default_matrix[3][3] = {{1.1, 1.2, 1.3}, {2.1, 2.2, 2.3}, {3.1, 3.2, 3.3}};

class configurableCut
{
 public:
  configurableCut(float cut_ = 2., int state_ = 1, bool option_ = true,
                  std::vector<float> bins_ = {0.5, 1.5, 2.5},
                  std::vector<std::string> labels_ = {"l1", "l2", "l3"},
                  o2::framework::Array2D<double> cuts_ = {&default_matrix[0][0], 3, 3})
    : cut{cut_}, state{state_}, option{option_}, bins{bins_}, labels{labels_}, cuts{cuts_}
  {
  }

  bool method(float arg) const;

  void setCut(float cut_);
  float getCut() const;

  void setState(int state_);
  int getState() const;

  void setOption(bool option_);
  bool getOption() const;

  void setBins(std::vector<float> bins_);
  std::vector<float> getBins() const;

  void setLabels(std::vector<std::string> labels_);
  std::vector<std::string> getLabels() const;

  void setCuts(o2::framework::Array2D<double> cuts_);
  o2::framework::Array2D<double> getCuts() const;

 private:
  float cut;
  int state;
  bool option;
  std::vector<float> bins;
  std::vector<std::string> labels;
  o2::framework::Array2D<double> cuts;

  ClassDef(configurableCut, 5);
};

std::ostream& operator<<(std::ostream& os, configurableCut const& c);

#endif // CONFIGURABLECUT_H
