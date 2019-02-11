#ifndef ALIVTPCSEED_H
#define ALIVTPCSEED_H

/* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
 * See cxx source for full Copyright notice                               *
 * Primary Author: Mikolaj Krzewicki, mkrzewic@cern.ch
 */
class AliTPCseed;

class AliVTPCseed {
  public:
  AliVTPCseed() {}
  virtual ~AliVTPCseed() {}
  virtual void CopyToTPCseed( AliTPCseed &) const = 0;
  virtual void SetFromTPCseed( const AliTPCseed*) = 0;
  //
  // special method for eventual suppression of shared clusters before deletion
  virtual void TagSuppressSharedClusters() {}
};

#endif
