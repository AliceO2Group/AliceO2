/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOIO_H
#define FAIRGEOIO_H

#include <Rtypes.h>    // for Bool_t, FairGeoIo::Class, etc
#include <TObject.h>   // for TObject

class FairGeoMedia;
class FairGeoSet;
class FairGeoInterface;

/**
 * Abstract base class for geometry I/O
 * @author Ilse Koenig
 */
class FairGeoIo : public TObject
{
  public:
    virtual Bool_t open(const char*, const Text_t* status = "in") = 0;
    virtual void close() = 0;
    virtual void print() = 0;
    virtual Bool_t isOpen() = 0;
    virtual Bool_t isWritable() = 0;
    virtual Bool_t read(FairGeoMedia*) = 0;
    virtual Bool_t read(FairGeoSet*, FairGeoMedia*) = 0;
    virtual Bool_t write(FairGeoMedia*) = 0;
    virtual Bool_t write(FairGeoSet*) = 0;
    virtual Bool_t readGeomConfig(FairGeoInterface*) = 0;
    virtual Bool_t setSimulRefRun(const char*) = 0;
    virtual Bool_t setHistoryDate(const char*) = 0;

  private:
    ClassDef(FairGeoIo, 0);
};

#endif /* !FAIRGEOIO_H */
