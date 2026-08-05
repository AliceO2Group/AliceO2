/********************************************************************************
 *    Copyright (C) 2014 GSI Helmholtzzentrum fuer Schwerionenforschung GmbH    *
 *                                                                              *
 *              This software is distributed under the terms of the             *
 *              GNU Lesser General Public Licence (LGPL) version 3,             *
 *                  copied verbatim in the file "LICENSE"                       *
 ********************************************************************************/
#ifndef FAIRGEOMEDIA_H
#define FAIRGEOMEDIA_H

#include <Rtypes.h>    // for FairGeoMedia::Class, etc
#include <TNamed.h>    // for TNamed
#include <TString.h>   // for TString
#include <iosfwd>      // for fstream

class FairGeoMedium;
class TList;

/**
 * Class for materials and media
 * @author Ilse koenig
 */
class FairGeoMedia : public TNamed
{
  private:
    TList* media;        /** list of media */
    TString inputFile;   /** name of input file or Oracle */
    TString author;      /** author of the media version */
    TString description; /** description of the version */
  public:
    FairGeoMedia();
    ~FairGeoMedia();
    void setInputFile(const char* file) { inputFile = file; }
    const char* getInputFile() { return inputFile; }
    void addMedium(FairGeoMedium* m);
    FairGeoMedium* getMedium(const char*);
    TList* getListOfMedia() { return media; }
    void read(std::fstream&);
    void print();
    void list();
    void write(std::fstream&);
    void setAuthor(TString& s) { author = s; }
    void setDescription(TString& s) { description = s; }
    TString& getAuthor() { return author; }
    TString& getDescription() { return description; }

  private:
    FairGeoMedia(const FairGeoMedia&);
    FairGeoMedia& operator=(const FairGeoMedia&);

    ClassDef(FairGeoMedia, 0);
};

#endif /* !FAIRGEOMEDIA_H */
