// Copyright CERN and copyright holders of ALICE O2. This software is
// distributed under the terms of the GNU General Public License v3 (GPL
// Version 3), copied verbatim in the file "COPYING".
//
// See http://alice-o2.web.cern.ch/license for full licensing information.
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

//#include "TRDBase/TRDGeometryBase.h"
//#include "DetectorsCommonDataFormats/DetMatrixCache.h"
//#include "DetectorsCommonDataFormats/DetID.h"

#ifndef O2_TRDTRACKLETBASE_H
#define O2_TRDTRACKLETBASE_H

////////////////////////////////////////////////////////////////////////////
//                                                                        //
// TRD tracklet                                                           //
// abstract base class for TRD tracklets                                  //
//                                                                        //
// Authors                                                                //
//  Alex Bercuci (A.Bercuci@gsi.de)                                       //
//  Jochen Klein (jochen.klein@cern.ch)                                   //
//                                                                        //
////////////////////////////////////////////////////////////////////////////
namespace o2
{
namespace trd
{

//TODO check with ?? I think Ole, about this as I think my freedom to
//define my own thing is limited by his requirements.

class TrackletBase
{

 public:
  TrackletBase() = default;
  TrackletBase(const TrackletBase& o) {}
  ~TrackletBase() = default;

  TrackletBase& operator=(const TrackletBase& o) { return *this; }

  virtual bool cookPID() = 0;

  virtual int getDetector() const = 0;
  virtual int getHCId() const { return 2 * getDetector() + (getYbin() > 0 ? 1 : 0); }

  virtual float getX() const = 0;
  virtual float getY() const = 0;
  virtual float getZ() const = 0;
  virtual float getdYdX() const = 0;
  virtual float getdZdX() const { return 0; }

  virtual int getdY() const = 0;   // in units of 140um
  virtual int getYbin() const = 0; // in units of 160um
  virtual int getZbin() const = 0; // in pad length units

  virtual double getPID(int is = -1) const = 0;

  virtual void localToGlobal(float&, float&, float&, float&) {}

  virtual void print(std::string* /*option=""*/) const {}

  virtual unsigned int getTrackletWord() const = 0;

  virtual void setDetector(int id) = 0;

 protected:
};
} //namespace trd
} //namespace o2
#endif
