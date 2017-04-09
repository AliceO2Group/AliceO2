#include "CCDB/ConditionId.h"
#include <TObjArray.h>    // for TObjArray
#include <TObjString.h>   // for TObjString
#include <Riostream.h>
// using std::endl;
// using std::cout;
using namespace o2::CDB;

ClassImp(ConditionId)

ConditionId::ConditionId() : mPath(), mIdRunRange(-1, -1), mVersion(-1), mSubVersion(-1), mLastStorage("new")
{
  // constructor
}

ConditionId::ConditionId(const ConditionId &other)
  : TObject(),
    mPath(other.mPath),
    mIdRunRange(other.mIdRunRange),
    mVersion(other.mVersion),
    mSubVersion(other.mSubVersion),
    mLastStorage(other.mLastStorage)
{
  // constructor
}

ConditionId::ConditionId(const IdPath &path, Int_t firstRun, Int_t lastRun, Int_t version, Int_t subVersion)
  : mPath(path), mIdRunRange(firstRun, lastRun), mVersion(version), mSubVersion(subVersion), mLastStorage("new")
{
  // constructor
}

ConditionId::ConditionId(const IdPath &path, const IdRunRange &runRange, Int_t version, Int_t subVersion)
  : mPath(path), mIdRunRange(runRange), mVersion(version), mSubVersion(subVersion), mLastStorage("new")
{
  // constructor
}

ConditionId *ConditionId::makeFromString(const TString &idString)
{
  // constructor from string
  // string has the format as the output of ConditionId::ToString:
  // path: "TRD/Calib/PIDLQ"; run range: [0,999999999]; version: v0_s0

  ConditionId *id = new ConditionId("a/b/c", -1, -1, -1, -1);

  TObjArray *arr1 = idString.Tokenize(';');
  TIter iter1(arr1);
  TObjString *objStr1 = nullptr;
  while ((objStr1 = dynamic_cast<TObjString *>(iter1.Next()))) {
    TString buff(objStr1->GetName());

    if (buff.Contains("path:")) {
      TString path(buff(buff.First('\"') + 1, buff.Length() - buff.First('\"') - 2));
      id->setPath(path.Data());

    } else if (buff.Contains("run range:")) {
      TString firstRunStr(buff(buff.Index('[') + 1, buff.Index(',') - buff.Index('[') - 1));
      TString lastRunStr(buff(buff.Index(',') + 1, buff.Index(']') - buff.Index(',') - 1));
      id->setIdRunRange(firstRunStr.Atoi(), lastRunStr.Atoi());

    } else if (buff.Contains("version:")) {
      if (buff.Contains("_s")) {
        TString versStr(buff(buff.Last('v') + 1, buff.Index('_') - buff.Last('v') - 1));
        TString subVersStr(buff(buff.Last('s') + 1, buff.Length() - buff.Last('s') - 1));
        id->setVersion(versStr.Atoi());
        id->setSubVersion(subVersStr.Atoi());
      } else {
        TString versStr(buff(buff.Last('v') + 1, buff.Length() - buff.Last('v') - 1));
        id->setVersion(versStr.Atoi());
      }
    }
  }

  delete arr1;

  return id;
}

ConditionId::~ConditionId()
{
  // destructor
}

Bool_t ConditionId::isValid() const
{
  // validity check

  if (!(mPath.isValid() && mIdRunRange.isValid())) {
    return kFALSE;
  }

  // FALSE if doesn't have version but has subVersion
  return !(!hasVersion() && hasSubVersion());
}

Bool_t ConditionId::isEqual(const TObject *obj) const
{
  // check if this id is equal to other id (compares path, run range, versions)

  if (this == obj) {
    return kTRUE;
  }

  if (ConditionId::Class() != obj->IsA()) {
    return kFALSE;
  }
  ConditionId *other = (ConditionId *) obj;
  return mPath.getPathString() == other->getPathString() && mIdRunRange.isEqual(&other->getIdRunRange()) &&
         mVersion == other->getVersion() && mSubVersion == other->getSubVersion();
}

TString ConditionId::ToString() const
{
  // returns a string of ConditionId data

  TString result = Form(R"(path: "%s"; run range: [%d,%d])", getPathString().Data(), getFirstRun(), getLastRun());

  if (getVersion() >= 0) {
    result += Form("; version: v%d", getVersion());
  }
  if (getSubVersion() >= 0) {
    result += Form("_s%d", getSubVersion());
  }
  return result;
}

void ConditionId::print(Option_t * /*option*/) const
{
  // Prints ToString()

  std::cout << ToString().Data() << std::endl;
}

Int_t ConditionId::Compare(const TObject *obj) const
{
  //
  // compare according y
  ConditionId *o2 = (ConditionId *) obj;
  return TString(this->getPathString()).CompareTo((o2->getPathString()));
}

Bool_t ConditionId::IsSortable() const
{
  return kTRUE;
}
