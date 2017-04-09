/// \file Condition.h
/// \brief Implementation of the Condition class (CDB object) containing the condition and its metadata

#include "CCDB/Condition.h"
#include <FairLogger.h>  // for LOG
#include <cstddef>      // for NULL

namespace o2 { namespace CDB { class IdRunRange; }}

using namespace o2::CDB;

ClassImp(Condition)

Condition::Condition() : mObject(nullptr), mId(), mConditionMetaData(nullptr), mOwner(kFALSE)
{
}

Condition::Condition(TObject *object, const ConditionId &id, ConditionMetaData *metaData, Bool_t owner)
  : mObject(object), mId(id), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, ConditionMetaData *metaData,
                     Bool_t owner)
  : mObject(object), mId(path, runRange, -1, -1), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, Int_t version,
                     ConditionMetaData *metaData,
                     Bool_t owner)
  : mObject(object), mId(path, runRange, version, -1), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion,
                     ConditionMetaData *metaData, Bool_t owner)
  : mObject(object), mId(path, runRange, version, subVersion), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, ConditionMetaData *metaData,
                     Bool_t owner)
  : mObject(object), mId(path, firstRun, lastRun, -1, -1), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version,
                     ConditionMetaData *metaData,
                     Bool_t owner)
  : mObject(object), mId(path, firstRun, lastRun, version, -1), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::Condition(TObject *object, const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version,
                     Int_t subVersion,
                     ConditionMetaData *metaData, Bool_t owner)
  : mObject(object), mId(path, firstRun, lastRun, version, subVersion), mConditionMetaData(metaData), mOwner(owner)
{
  mConditionMetaData->setObjectClassName(mObject->ClassName());
}

Condition::~Condition()
{

  if (mOwner) {
    if (mObject) {
      delete mObject;
    }

    if (mConditionMetaData) {
      delete mConditionMetaData;
    }
  }
}

void Condition::printId() const
{

  LOG(INFO) << mId.ToString().Data() << FairLogger::endl;
}

Int_t Condition::Compare(const TObject *obj) const
{
  Condition *o2 = (Condition *) obj;
  return TString(this->getId().getPathString()).CompareTo((o2->getId().getPathString()));
}

Bool_t Condition::IsSortable() const
{
  return kTRUE;
}
