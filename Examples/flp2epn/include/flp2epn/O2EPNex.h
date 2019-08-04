// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

/**
 * O2EPNex.h
 *
 * @since 2013-01-09
 * @author D. Klein, A. Rybalchenko, M.Al-Turany
 */

#ifndef O2EPNEX_H_
#define O2EPNEX_H_

#include <FairMQDevice.h>

class O2EPNex : public FairMQDevice
{
 public:
  O2EPNex();

  ~O2EPNex() override;

  bool Process(std::unique_ptr<FairMQMessage>&, int);
};

#endif
