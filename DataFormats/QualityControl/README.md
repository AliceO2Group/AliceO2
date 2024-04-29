\page refDataFormatsQualityControl Data Formats Quality Control

## Tagging Good Quality Data for Processing and Analysis

This document outlines the data formats used for tagging good quality data for further processing and analysis.
They allow us to describe problems affecting the data in concrete time intervals.
Using this information, data can be filtered out according to criteria specific to a given analysis type.

### Data Quality Control workflow

Data quality is determined through two methods:

1. **Automated Checks:** The Quality Control framework runs Checks which may return quality Flags.
2. **Manual Review:** Detector experts review data using the Run Condition Table (RCT) and can modify or add Flags.

Both methods utilize the same data format for Flags.
During processing (both synchronous and asynchronous), Checks produce Qualities and associate them with Flags.
The Quality Control framework then transmits these Flags to the RCT through a gRPC interface.
Detector experts can then review the automatically generated Flags and make any necessary modifications or additions directly in the RCT.

### Quality Control Flag Structure

A [Quality Control Flag](include/DataFormatsQualityControl/QualityControlFlag.h) consists of the following elements:

* **Flag Type:** This identifies the specific issue the flag represents. (More details below)
* **Time Range:** This defines the time period affected by the flag.
* **Comment (Optional):** This allows human-readable explanations for assigning the flag.
* **Source String:** This identifies the entity (person or software module) that created the flag.

**Example:**

```
flag: Limited Acceptance MC Reproducible
start: 1612707603626
end: 1613999652000
comment: Sector B in TPC inactive
source: TPC/Clusters Check
```

### Flag Types

[Flag Types](include/DataFormatsQualityControl/FlagType.h) define the specific categories of issues represented by flags.
Each Flag Type has the following attributes:

* **Identifier Number:** A unique numerical ID for the Flag Type.
* **Name:** A human-readable name describing the issue.
* **"Bad Quality" Determinant:** This boolean constant indicates whether the type inherently signifies bad data quality.

#### Creating and Managing Flag Types

* **FlagTypeFactory** ensures a centralized and consistent list of available Flag Types.
  New types can only be created through this factory.
* **[flagTypes.csv](etc/flagTypes.csv)** defines the existing Flag Types, including their ID, name, and "bad quality" determinant, factory method name and a switch to deprecate a flag.
  The table serves as the source to automatically generate the corresponding methods in FlagTypeFactory.
* **Adding new Flag Types:** If a new issue requires a flag not currently defined, propose the addition by contacting the async QC coordinators.
  They have the authority to add new Flag Types to the RCT.
  These changes will then be reflected in the [flagTypes.csv](etc/flagTypes.csv) file through a pull request.
* **Modification of existing Flag Types:** Existing Flag Types should not be modified in terms of their definition.
  Instead, one may create a new Flag Type and mark the existing one as obsolete in the CSV table.
  This will add the `[[ deprecated ]]` attribute to the corresponding method.

#### Currently available Flag Types

This section details the currently available Flag Types and provides a brief explanation of their intended use cases.

* **Good:** a Check or an expert sees nothing wrong with given time interval, but would like to add a comment.
  Note that the absence of any flag for a run implies good data quality.
  Thus, there is no need to mark it explicitly as such by using this flag type.
* **No Detector Data:** a complete and unexpected absence of data for a specific detector.
* **Limited Acceptance MC Not Reproducible:** a part of a detector did not acquire good data and this condition cannot be reproduced in Monte Carlo.
  If an automated Check cannot determine MC reproducibility, it should default to "Not Reproducible" for later expert review.
* **Limited Acceptance MC Reproducible:** a part of a detector did not acquire good data, but this condition can be reproduced in Monte Carlo.
* **Bad Tracking:** analyses relying on accurate track reconstruction should not use this data.
* **Bad PID:** analyses relying on correct identification of all kinds of tracked particles should not use this data.
* **Bad Hadron PID:** analyses relying on correct hadron identification should not use this data.
* **Bad Electron PID:** analyses relying on correct electron identification should not use this data.
* **Bad Photon Calorimetry:** analyses relying on correct photon calorimetry should not use this data.
* **Bad EMCalorimetry:** analyses relying on correct electromagnetic calorimetry should not use this data.
* **Unknown:** the exact impact of an issue on the data is unclear, but it's likely bad.
  Treat data with this flag with caution until further investigation.
* **Unknown Quality:** the quality of data could not be determined definitively.
* **Invalid:** there was an issue with processing the flags.

## Usage in Analysis framework (plans, wishlist)

## General idea
* RCT exports a read-only copy of the flags for each detector, run and pass combination in the CCDB.
* The Analysis framework uses the flags and user selection criteria to provide their Analysis Task with data matching these criteria.
  **Data Tags** are the structures defining the good time intervals for the provided criteria.

## Notes on plans for Data Tags

Data Tag definition has:
* name
* global suppression list for bad flag types (applies to all detectors)
* global requirement list for not bad flag types
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