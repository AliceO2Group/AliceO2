// Copyright 2019-2020 CERN and copyright holders of ALICE O2.
// See https://alice-o2.web.cern.ch/copyright for details of the copyright holders.
// All rights not expressly granted are reserved.
//
// This software is distributed under the terms of the GNU General Public
// License v3 (GPL Version 3), copied verbatim in the file "COPYING".
//
// In applying this license CERN does not waive the privileges and immunities
// granted to it by virtue of its status as an Intergovernmental Organization
// or submit itself to any jurisdiction.

#ifndef O2_CCDB_RESPONSE_TEST_RESOURCES_H
#define O2_CCDB_RESPONSE_TEST_RESOURCES_H

namespace o2
{
namespace ccdb
{

const char* firstFullResponse = R"({
    "objects": [
        {
            "path": "Users/g/grigoras/testing/grid/a",
            "createTime": 1637685278841,
            "lastModified": 1637685278841,
            "Created": 1637685278841,
            "Last-Modified": 1637685278841,
            "id": "407f3a65-4c7b-11ec-8cf8-200114580202",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"407f3a65-4c7b-11ec-8cf8-200114580202\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "2001:1458:202:56::101:bc25",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "2001:1458:202:56:0:0:101:bc25",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "/download/407f3a65-4c7b-11ec-8cf8-200114580202"
            ]
        },
        {
            "path": "Users/g/grigoras/testing/grid/b",
            "createTime": 1637678437647,
            "lastModified": 1637678437647,
            "Created": 1637678437647,
            "Last-Modified": 1637678437647,
            "id": "52d3f61a-4c6b-11ec-a98e-7f000001aa8b",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"52d3f61a-4c6b-11ec-a98e-7f000001aa8b\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "127.0.0.1",
            "preservation": "false",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "127.0.0.1",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "/download/52d3f61a-4c6b-11ec-a98e-7f000001aa8b"
            ]
        }
    ]
}
)";

const char* secondFullResponse = R"({
    "objects": [
        {
            "path": "Users/g/grigoras/testing/grid/a",
            "createTime": 1637685278841,
            "lastModified": 1637685278841,
            "Created": 1637685278841,
            "Last-Modified": 1637685278841,
            "id": "407f3a65-4c7b-11ec-8cf8-200114580202",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"407f3a65-4c7b-11ec-8cf8-200114580202\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "2001:1458:202:56::101:bc25",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "2001:1458:202:56:0:0:101:bc25",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "/download/407f3a65-4c7b-11ec-8cf8-200114580202"
            ]
        },
        {
            "path": "Users/g/grigoras/testing/grid/a",
            "createTime": 1637685125493,
            "lastModified": 1637685125493,
            "Created": 1637685125493,
            "Last-Modified": 1637685125493,
            "id": "e5183d1a-4c7a-11ec-9d71-7f000001aa8b",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"e5183d1a-4c7a-11ec-9d71-7f000001aa8b\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "127.0.0.1",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "127.0.0.1",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "alien:///alice/data/CCDB/Users/g/grigoras/testing/grid/00/18200/e5183d1a-4c7a-11ec-9d71-7f000001aa8b",
                "/download/e5183d1a-4c7a-11ec-9d71-7f000001aa8b"
            ]
        },
        {
            "path": "Users/g/grigoras/testing/grid/a",
            "createTime": 1637678437647,
            "lastModified": 1637678437647,
            "Created": 1637678437647,
            "Last-Modified": 1637678437647,
            "id": "52d3f61a-4c6b-11ec-a98e-7f000001aa8b",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"52d3f61a-4c6b-11ec-a98e-7f000001aa8b\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "127.0.0.1",
            "preservation": "false",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "127.0.0.1",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "/download/52d3f61a-4c6b-11ec-a98e-7f000001aa8b"
            ]
        },
        {
            "path": "Users/g/grigoras/testing/grid/c",
            "createTime": 1637678437647,
            "lastModified": 1637678437647,
            "Created": 1637678437647,
            "Last-Modified": 1637678437647,
            "id": "99d3f61a-4c6b-11ec-a98e-7f000001aa8b",
            "validFrom": 1604175160884,
            "validUntil": 1919535160884,
            "initialValidity": 1919535160884,
            "MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "fileName": "TObject_1595938437506.root",
            "contentType": "application/octet-stream",
            "size": 4096,
            "ETag": "\"52d3f61a-4c6b-11ec-a98e-7f000001aa8b\"",
            "Valid-From": 1604175160884,
            "Valid-Until": 1919535160884,
            "InitialValidityLimit": 1919535160884,
            "Content-MD5": "c1ffddcd4db30e3e684180165acd63ca",
            "Content-Disposition": "inline;filename=\"TObject_1595938437506.root\"",
            "Content-Type": "application/octet-stream",
            "Content-Length": 4096,
            "UploadedFrom": "127.0.0.1",
            "preservation": "false",
            "ObjectType": "TH1F",
            "qc_detector_name": "HMP",
            "custom": "34",
            "UpdatedFrom": "127.0.0.1",
            "qc_task_name": "daqTask",
            "qc_version": "1.1.0",
            "partName": "send",
            "replicas": [
                "/download/52d3f61a-4c6b-11ec-a98e-7f000001aa8b"
            ]
        }
    ]
}
)";

const char* emptyResponse = R"({
    "objects": [],
})";

} // namespace ccdb
} // namespace o2

#endif