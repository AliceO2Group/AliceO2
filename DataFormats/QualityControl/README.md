\page refDataFormatsQualityControl Data Formats Quality Control

Data formats for tagging good quality data for analysis.

# Flagging time ranges
## General idea
* Each detector has its own CCDB (QCDB) entry - TimeRangeFlagCollection
* The CCDB Entry validity will be run or fill, depending on the final data taking granularity
* Each entry can define sub ranges (TimeRangeFlags) of certain data characteristics (FlagReasons) inside the CCDB Entry validity
* Flags are defined in a common store, they are used to derive the Data Tags - the final filters for good quality data during analysis.
* Data Tag are stored as CCDB entries. They might require different detectors and may suppress different flags dependent on the analysis type.

## Implementation

[Flag Reason](include/DataFormatsQualityControl/FlagReasons.h) is defined with an identifier number, a name and a 'bad quality' determinant.
The latter decides if such a flag should mark the data quality as bad by default.
FlagReasons can be created only with FlagReasonFactory, so that the list of available reasons is common and centralised. 
For example:
```
id: 10
name: Limited Acceptance
bad: true
```

With [TimeRangeFlags](include/DataFormatsQualityControl/TimeRangeFlag.h) we can define the time range of a chosen FlagReason, add an additional comment and specify the source of this flag.
For example:
```
start: 1612707603626 
end: 1613999652000
flag: Limited Acceptance
comment: Sector B in TPC inactive
source: o2::quality_control_modules::tpc::ClustersCheck
```

[TimeRangeFlagCollection](include/DataFormatsQualityControl/TimeRangeFlagCollection.h) contains all TimeRangeFlags for the validity range (run or fill).
TimeRangeFlags may overlap, e.g. if they use different FlagReasons and they are sorted by their start time.
If certain period does not include any TimeRangeFlags with *bad* FlagReasons, then the data quality can be considered as good.
The [TimeRangeFlagCollection test](test/testTimeRangeFlagCollection.cxx) shows the usage example.

TimeRangeFlagCollections are supposed to be created automatically with QC Post-processing Tasks based on Quality Objects created by QC Checks.
However, they might be created manually by QA experts as well.
The procedure to do that has to be defined.

## TODO
* Define the complete list of available Flag Reasons
* implement CCDB storage and access
  - define CCDB storage place e.g.
    * `<Detector>/QC/QualityFlags`
    * `Analysis/QualityFlags/<Detector>`
* Data Tags Definitions and Data Tags

### Notes on plans for Data Tags

Data Tag definition has:
* name
* global suppression list for bad flag reasons (applies to all detectors)
* global requirement list for not bad flag reasons
* list of needed detectors,
* list of suppression list and requirement list specific to detectors

Example configurations
```json
{
  "name": "CentralBarrelTrackingLimitedAcceptance",
  "globalRequiredDetectors" : [ "TPC", "ITS" ],
  "globalIgnoredFlags" : ["Limited acceptance", "asdf"],
  "globalRequiredFlags" : [],
  "detectorSpecificFlags" : [
    {
      "name" : "TPC",
      "ignoredFlags" : [ "Reason X" ]
    }
  ]
},
{
  "name": "CentralBarrelTracking",
  "globalRequiredDetectors" : [ "TPC", "ITS" ],
  "globalIgnoredFlags" : [],
  "globalRequiredFlags" : [],
  "detectorSpecificFlags" : []
},
{
  "name": "CentralBarrelTrackingOnlyLimitedAcceptanceinTPC",
  "globalRequiredDetectors" : [ "TPC", "ITS" ],
  "globalIgnoredFlags" : [],
  "detectorSpecificFlags" : [
   {
      "name" : "TPC",
      "requiredFlags" : [ "Limited acceptance" ]
   }
}
]
```

## Wishlist / Ideas
* executable to add flags to the flag store
* executable to extrags the flag store
* summary of all masks for one TimeRangeFlagCollection
* functionality to extract flags for a specific detector (from CCDB)
* cut class to specify detectors and flags which to exclude