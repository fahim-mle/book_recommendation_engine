const request_to_API = "https://api.trove.nla.gov.au/v3/result?q=isbn:9781741250879&category=book&encoding=json"
const response_from_API={
    "query": "isbn:9781741250879",
    "category": [
        {
            "code": "book",
            "name": "Books & Libraries",
            "records": {
                "s": "*",
                "n": 2,
                "total": 2,
                "work": [
                    {
                        "id": "17911947",
                        "url": "https://api.trove.nla.gov.au/v3/work/17911947",
                        "troveUrl": "https://trove.nla.gov.au/work/17911947",
                        "title": "Victorian targeting handwriting /  Tricia Dearborn, Jo Ryan, Stephen Michael King",
                        "contributor": [
                            "Dearborn, Tricia"
                        ],
                        "issued": "2004",
                        "type": [
                            "Book",
                            "Book/Illustrated"
                        ],
                        "hasCorrections": "N",
                        "relevance": {
                            "score": 8.119243621826172,
                            "value": "very relevant"
                        },
                        "holdingsCount": 6,
                        "versionCount": 3
                    },
                    {
                        "id": "189779979",
                        "url": "https://api.trove.nla.gov.au/v3/work/189779979",
                        "troveUrl": "https://trove.nla.gov.au/work/189779979",
                        "title": "Targeting handwriting. : Victorian modern cursive / by Jane Pinsker and Jo Ryan ; illustrated by Stephen Michael King",
                        "contributor": [
                            "Pinsker, Jane"
                        ],
                        "issued": "2003-2004",
                        "type": [
                            "Book/Illustrated",
                            "Book"
                        ],
                        "hasCorrections": "N",
                        "relevance": {
                            "score": 8.119243621826172,
                            "value": "very relevant"
                        },
                        "holdingsCount": 4,
                        "versionCount": 17
                    }
                ]
            }
        }
    ]
}
const important_features = [id, title, contributor, issued, type, relevance, identifier]
// Below is the list of actual column names that will be used in the table. These names are derived from the API response and are used to map the data to the table structure.
const actual_column_name= ["trove_id", "title", "author", "issued", "type", "relevance_score", "relevance_value", "identifier_type", "identifier_linktype", "identifier_value"]