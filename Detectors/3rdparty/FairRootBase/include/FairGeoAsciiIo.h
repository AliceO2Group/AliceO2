/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOASCIIIO_H
#define FAIRGEOASCIIIO_H

#include "FairGeoIo.h"   // for FairGeoIo

#include <Rtypes.h>    // for Bool_t, etc
#include <TString.h>   // for TString
#include <iosfwd>      // for fstream

class FairGeoSet;
class FairGeoMedia;
class FairGeoInterface;

/**
 * Class for geometry I/O from ASCII file
 * @author Ilse Koenig
 */
class FairGeoAsciiIo : public FairGeoIo
{
    TString filename;
    TString filedir;
    Bool_t writable;
    std::fstream* file;

  public:
    FairGeoAsciiIo();
    virtual ~FairGeoAsciiIo();
    void setDirectory(const char* fDir) { filedir = fDir; }
    const char* getDirectory() { return filedir.Data(); }
    const char* getFilename() { return filename.Data(); }
    Bool_t open(const char*, const Text_t* status = "in");
    Bool_t isOpen();
    Bool_t isWritable();
    void close();
    void print();
    Bool_t read(FairGeoMedia*);
    Bool_t read(FairGeoSet*, FairGeoMedia*);
    Bool_t write(FairGeoMedia*);
    Bool_t write(FairGeoSet* set);
    Bool_t readGeomConfig(FairGeoInterface*);
    Bool_t readDetectorSetup(FairGeoInterface*);
    Bool_t setSimulRefRun(const char*) { return kTRUE; }
    Bool_t setHistoryDate(const char*) { return kTRUE; }

  private:
    FairGeoAsciiIo(const FairGeoAsciiIo&);
    FairGeoAsciiIo& operator=(const FairGeoAsciiIo&);
    ClassDef(FairGeoAsciiIo, 0);   // Class for geometry I/O from ASCII file
};

#endif /* !FAIRGEOASCIIIO_H */
