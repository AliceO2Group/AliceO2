/// \file Condition.h
/// \brief Definition of the Condition class (CDB object) containing the condition and its metadata

#ifndef ALICEO2_CDB_ENTRY_H_
#define ALICEO2_CDB_ENTRY_H_

#include "CCDB/ConditionId.h"        // for ConditionId
#include "CCDB/ConditionMetaData.h"  // for ConditionMetaData
#include "CCDB/IdPath.h"             // for IdPath
#include "Rtypes.h"             // for Int_t, kFALSE, Bool_t, etc
#include "TObject.h"            // for TObject
#include "TString.h"            // for TString

namespace o2 { namespace CDB { class IdRunRange; }}


namespace o2 {
namespace CDB {

/// Class containing the condition (a ROOT TObject) and the metadata identifying it (ConditionId)
/// An instance of this class is a CDB object and has a specified run-range validity, version
/// and subversion. These metadata are both building the file name of the CDB object and are
/// contained in the metadata.
class Condition : public TObject
{

  public:
    /// Default constructor
    Condition();

    /// Constructor
    Condition(TObject *object, const ConditionId &id, ConditionMetaData *metaData, Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, ConditionMetaData *metaData,
              Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, Int_t version,
              ConditionMetaData *metaData,
              Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion,
              ConditionMetaData *metaData, Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, ConditionMetaData *metaData,
              Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version,
              ConditionMetaData *metaData,
              Bool_t owner = kFALSE);

    /// Constructor
    Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version, Int_t subVersion,
              ConditionMetaData *metaData, Bool_t owner = kFALSE);

    /// Default destructor
    ~Condition() override;

    /// Set the object identity from an ConditionId
    void setId(const ConditionId &id)
    {
      mId = id;
    };

    /// ConditionId accessor
    ConditionId &getId()
    {
      return mId;
    };

    /// ConditionId accessor
    const ConditionId &getId() const
    {
      return mId;
    };

    /// Print the identifier fields
    void printId() const;

    /// Setter of the TObject data member
    void setObject(TObject *object)
    {
      mObject = object;
    };

    /// Getter of the TObject data member
    TObject *getObject()
    {
      return mObject;
    };

    /// Getter of the TObject data member
    const TObject *getObject() const
    {
      return mObject;
    };

    void setConditionMetaData(ConditionMetaData *metaData)
    {
      mConditionMetaData = metaData;
    };

    ConditionMetaData *getConditionMetaData()
    {
      return mConditionMetaData;
    };

    const ConditionMetaData *getConditionMetaData() const
    {
      return mConditionMetaData;
    };

    void printConditionMetaData() const
    {
      mConditionMetaData->printConditionMetaData();
    }

    /// Getter of the TObject data member
    void setOwner(Bool_t owner)
    {
      mOwner = owner;
    };

    /// Getter of the TObject data member
    Bool_t isOwner() const
    {
      return mOwner;
    };

    /// Setter of the version
    void setVersion(Int_t version)
    {
      mId.setVersion(version);
    }

    /// Setter of the subversion
    void setSubVersion(Int_t subVersion)
    {
      mId.setSubVersion(subVersion);
    }

    /// Getter of the last storage where the CDB object was before the current one
    const TString getLastStorage() const
    {
      return mId.getLastStorage();
    };

    /// Getter of the last storage where the CDB object was before the current one
    void setLastStorage(TString lastStorage)
    {
      mId.setLastStorage(lastStorage);
    };

    /// Method to compare two CDB objects, used for sorting in ROOT ordered containers
    Int_t Compare(const TObject *obj) const override;

    /// Define this class sortable (via the Compare method) for ROOT ordered containers
    Bool_t IsSortable() const override;

  private:
    Condition(const Condition &other);          // no copy ctor
    void operator=(const Condition &other); // no assignment op

    TObject *mObject;    ///< The actual condition, a ROOT TObject
    ConditionId mId;        ///< The condition identifier
    ConditionMetaData *mConditionMetaData; ///< metaData
    Bool_t mOwner;     ///< Ownership flag

  ClassDefOverride(Condition, 1)
};
}
}
#endif
