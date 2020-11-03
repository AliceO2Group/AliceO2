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

#include <iosfwd>
#include <TMath.h>

class configurableCut
{
 public:
  configurableCut(float cut_ = 2., int state_ = 1, bool option_ = true)
    : cut{cut_}, state{state_}, option{option_}
  {
  }

  bool method(float arg) const;

  void setCut(float cut_);
  float getCut() const;

  void setState(int state_);
  int getState() const;

  void setOption(bool option_);
  bool getOption() const;

 private:
  float cut;
  int state;
  bool option;

  ClassDef(configurableCut, 3);
};

std::ostream& operator<<(std::ostream& os, configurableCut const& c);

#endif // CONFIGURABLECUT_H
