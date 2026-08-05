/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
//*-- AUTHOR : Ilse Koenig
//*-- Created : 10/11/2003

///////////////////////////////////////////////////////////////////////////////
// FairGeoAsciiIo
//
// Class for geometry I/O from ASCII file
//
///////////////////////////////////////////////////////////////////////////////
#include "FairGeoAsciiIo.h"

#include "FairGeoInterface.h"   // for FairGeoInterface
#include "FairGeoIo.h"          // for FairGeoIo
#include "FairGeoMedia.h"       // for FairGeoMedia
#include "FairGeoSet.h"         // for FairGeoSet

#include <fstream>    // for fstream
#include <iostream>   // for cout
#include <stdio.h>    // for sscanf
#include <string.h>   // for strcmp, strlen, strtok, etc

using std::cout;
using std::endl;
using std::ios;

ClassImp(FairGeoAsciiIo);

FairGeoAsciiIo::FairGeoAsciiIo()
    : FairGeoIo()
    , filename("")
    , filedir("")
    , writable(kFALSE)
    , file(nullptr)
{
    // Constructor
}

FairGeoAsciiIo::~FairGeoAsciiIo()
{
    // Destructor
    close();
    delete file;
    file = 0;
}

Bool_t FairGeoAsciiIo::open(const char* fname, const Text_t* status)
{
    // Opens the file fname
    close();
    if (!file) {
        file = new std::fstream();
    } else {
        (file->clear());
    }
    if (!filedir.IsNull()) {
        filename = filedir + "/" + fname;
    } else {
        filename = fname;
    }
    filename = filename.Strip();
    if (strcmp(status, "in") == 0) {
        file->open(filename, ios::in);
        writable = kFALSE;
    } else {
        if (strcmp(status, "out") == 0) {
            file->open(filename, ios::in);
            if (!isOpen()) {
                file->close();
                file->clear();
                file->open(filename, ios::out);
                writable = kTRUE;
            } else {
                file->close();
                Error("open", "Output file %s exists already and will not be recreated.", filename.Data());
                return kFALSE;
            }
        } else {
            Error("open", "Invalid file option!");
        }
    }
    if (file->rdbuf()->is_open() == 0) {
        Fatal("open", "Failed to open file %s", filename.Data());
        return kFALSE;
    }
    return kTRUE;
}

Bool_t FairGeoAsciiIo::isOpen()
{
    // Returns kTRUE, if the file is open
    if (file && file->rdbuf()->is_open() == 1) {
        return kTRUE;
    }
    return kFALSE;
}

Bool_t FairGeoAsciiIo::isWritable()
{
    // Returns kTRUE, if the file is open and writable
    if (isOpen() && writable) {
        return kTRUE;
    }
    return kFALSE;
}

void FairGeoAsciiIo::close()
{
    // Closes the file
    if (isOpen()) {
        file->close();
        filename = "";
    }
}

void FairGeoAsciiIo::print()
{
    // Prints file information
    if (isOpen()) {
        if (writable) {
            cout << "Open output file: " << filename << endl;
        } else {
            cout << "Open input file: " << filename << endl;
        }
    } else {
        cout << "No file open." << endl;
    }
}

Bool_t FairGeoAsciiIo::read(FairGeoMedia* media)
{
    // Reads the media from file
    if (!isOpen() || writable || media == 0) {
        return kFALSE;
    }
    media->read(*file);
    return kTRUE;
}

Bool_t FairGeoAsciiIo::read(FairGeoSet* set, FairGeoMedia* media)
{
    // Reads the geometry set from file
    if (!isOpen() || writable || set == 0) {
        return kFALSE;
    }
    set->read(*file, media);
    return kTRUE;
}

Bool_t FairGeoAsciiIo::write(FairGeoMedia* media)
{
    // Writes the media to file
    if (!isOpen() || !writable || media == 0) {
        return kFALSE;
    }
    media->write(*file);
    return kTRUE;
}

Bool_t FairGeoAsciiIo::write(FairGeoSet* set)
{
    // Writes the geometry set to file
    if (!isOpen() || !writable || set == 0) {
        return kFALSE;
    }
    set->write(*file);
    return kTRUE;
}

Bool_t FairGeoAsciiIo::readGeomConfig(FairGeoInterface* interface)
{
    // Reads the GEANT configuration file
    if (!isOpen() || writable || interface == 0) {
        return kFALSE;
    }
    TString buf(256);
    TString simRefRun, historyDate;
    Int_t k = -1;
    while (!(*file).eof()) {
        buf.ReadLine(*file);
        buf = buf.Strip(buf.kBoth);
        if (!buf.IsNull() && buf(0, 2) != "//" && buf(0, 1) != "*") {
            if (buf.Contains(".geo") || buf.Contains("_gdb")) {
                interface->addInputFile(buf.Data());
            } else {
                if (buf.Contains(".setup")) {
                    interface->addSetupFile(buf.Data());
                } else {
                    if (buf.Contains("SimulRefRunDb:")) {
                        k = buf.Last(' ') + 1;
                        if (k) {
                            simRefRun = buf(k, buf.Length() - k);
                        }
                    } else {
                        if (buf.Contains("HistoryDateDb:")) {
                            k = buf.First(' ') + 1;
                            if (buf.Length() > k) {
                                historyDate = buf(k, buf.Length() - k);
                                historyDate = historyDate.Strip(historyDate.kBoth);
                            }
                        }
                    }
                }
            }
        }
    }
    Bool_t rc = kTRUE;
    FairGeoIo* oraIo = interface->getOraInput();
    if (oraIo) {
        if (historyDate.Length() > 0) {
            rc = oraIo->setHistoryDate(historyDate.Data());
        }
        if (rc && simRefRun.Length() > 0) {
            rc = oraIo->setSimulRefRun(simRefRun.Data());
        }
    }
    return rc;
}

Bool_t FairGeoAsciiIo::readDetectorSetup(FairGeoInterface* interface)
{
    // Reads the detector setups, needed for create only subsets
    if (!isOpen() || writable || interface == 0) {
        return kFALSE;
    }
    const Int_t maxbuf = 256;
    char buf[maxbuf];
    TString s, det;
    FairGeoSet* set = 0;
    Int_t maxModules = 0, secNo = -1;
    Int_t* mod = 0;
    const char d[] = " ";
    while (!(*file).eof()) {
        (*file).getline(buf, maxbuf);
        if (strlen(buf) >= 3 && buf[1] != '/') {
            if (buf[0] == '[') {
                set = 0;
                delete[] mod;
                mod = 0;
                s = buf;
                Ssiz_t n = s.First(']');
                det = s(1, n - 1);
                det.ToLower();
                set = interface->findSet(det);
                if (!set) {
                    Error("readDetectorSetup", "Detector %s not found", det.Data());
                    delete[] mod;
                    return kFALSE;
                }
                maxModules = set->getMaxModules();
                mod = new Int_t[maxModules];
            } else {
                if (set && mod) {
                    char* ss = strtok(buf, d);
                    if (ss && strlen(ss) > 3) {
                        secNo = static_cast<Int_t>(ss[3] - '0') - 1;
                        for (Int_t i = 0; i < maxModules && mod; i++) {
                            ss = strtok(nullptr, d);
                            if (ss) {
                                sscanf(ss, "%i", &mod[i]);
                            }
                        }
                        set->setModules(secNo, mod);
                    }
                } else {
                    delete[] mod;
                    return kFALSE;
                }
            }
        }
    }
    delete[] mod;
    return kTRUE;
}
