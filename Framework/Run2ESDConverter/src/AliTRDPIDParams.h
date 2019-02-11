/**************************************************************************
* Copyright(c) 1998-1999, ALICE Experiment at CERN, All rights reserved. *
*                                                                        *
* Author: The ALICE Off-line Project.                                    *
* Contributors are mentioned in the code where appropriate.              *
*                                                                        *
* Permission to use, copy, modify and distribute this software and its   *
* documentation strictly for non-commercial purposes is hereby granted   *
* without fee, provided that the above copyright notice appears in all   *
* copies and that both the copyright notice and this permission notice   *
* appear in the supporting documentation. The authors make no claims     *
* about the suitability of this software for any purpose. It is          *
* provided "as is" without express or implied warranty.                  *
**************************************************************************/
//
// Container for TRD thresholds stored in the OADB
//
#ifndef ALITRDPIDPARAMS_H
#define ALITRDPIDPARAMS_H

#ifndef ROOT_TNamed
#include <TNamed.h>
#endif

class TList;
class TSortedList;

class AliTRDPIDParams : public TNamed{
public:
    AliTRDPIDParams();
    AliTRDPIDParams(const char *name);
    AliTRDPIDParams(const AliTRDPIDParams &);
    virtual ~AliTRDPIDParams();
    virtual void Print(Option_t *) const;

    void AddCentralityClass(Double_t minCentrality, Double_t maxCentrality);
    Bool_t GetThresholdParameters(Int_t ntracklets, Double_t efficiency, Double_t *params, Double_t centrality = -1, Int_t charge=0) const;
    void SetThresholdParameters(Int_t ntracklets, Double_t effMin, Double_t effMax, Double_t *params, Double_t centrality = -1, Int_t charge=0);

    /* private: */
    class AliTRDPIDThresholds : public TObject{
    public:
        AliTRDPIDThresholds();
        AliTRDPIDThresholds(Int_t nTracklets, Double_t effMin, Double_t effMax, Double_t *params, Int_t charge);
        AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Double_t *params, Int_t charge);
        AliTRDPIDThresholds(Int_t nTracklets, Double_t effMin, Double_t effMax, Double_t *params = NULL);
        AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Double_t *params = NULL);
	AliTRDPIDThresholds(Int_t nTracklets, Double_t eff, Int_t charge);
        AliTRDPIDThresholds(const AliTRDPIDThresholds &);
        AliTRDPIDThresholds &operator=(const AliTRDPIDThresholds &);
        virtual ~AliTRDPIDThresholds() {}

        Int_t GetNTracklets() const { return fNTracklets; }
        Double_t GetElectronEfficiency(Int_t step = 0) const { if(step == 0) return fEfficiency[0]; else return fEfficiency[1]; }
        Int_t GetCharge() const {return fCharge;}
        const Double_t *GetThresholdParams() const { return fParams; }

        virtual Bool_t IsSortable() const { return kTRUE; }
        virtual Int_t Compare(const TObject *ref) const;
        virtual Bool_t IsEqual(const TObject *ref) const;

    private:
        Int_t fNTracklets;          //
        Double_t fEfficiency[2];    //
        Double_t fParams[4];        //
        Int_t fCharge;              //


        // #if ROOT_VERSION_CODE < ROOT_VERSION(5,99,0)
        // Private - cannot be streamed
        ClassDef(AliTRDPIDThresholds, 3);
        // #endif
    };

    class AliTRDPIDCentrality : public TObject{
    public:
        AliTRDPIDCentrality();
        AliTRDPIDCentrality(Double_t minCentrality, Double_t maxCentrality);
        AliTRDPIDCentrality(const AliTRDPIDCentrality &ref);
        AliTRDPIDCentrality &operator=(const AliTRDPIDCentrality &ref);
        virtual ~AliTRDPIDCentrality();

        Double_t GetMinCentrality() const { return fMinCentrality; };
        Double_t GetMaxCentrality() const { return fMaxCentrality; };
        
        void SetThresholdParameters(Int_t ntracklets, Double_t effMin, Double_t effMax, Double_t *params, Int_t charge=0);
        Bool_t GetThresholdParameters(Int_t ntracklets, Double_t efficiency, Double_t *params, Int_t charge=0) const;
        void Print(Option_t *) const;
    private:
        TSortedList *fEntries;       //
        Double_t fMinCentrality;     //
        Double_t fMaxCentrality;     //

        // #if ROOT_VERSION_CODE < ROOT_VERSION(5,99,0)
        // Private - cannot be streamed
        ClassDef(AliTRDPIDCentrality, 1);
        // #endif
    };
private:

    AliTRDPIDCentrality *FindCentrality(Double_t centrality) const;
    AliTRDPIDParams &operator=(const AliTRDPIDParams &);
    static const Double_t kVerySmall;
    TList *fEntries; //


    ClassDef(AliTRDPIDParams, 2);
};
#endif


