from unittest.mock import patch
# TODO: fix import *
from afl_api import *

TEST_COMPSEASON_API_RESPONSE = {
  "meta": {
    "code": 200,
    "pagination": {
      "page": 0,
      "numPages": 1,
      "pageSize": 14,
      "numEntries": 14
    }
  },
  "compSeasons": [
    {
      "id": 73,
      "providerId": "CD_S2025014",
      "name": "2025 Toyota AFL Premiership",
      "shortName": "Premiership",
      "currentRoundNumber": 0
    },
    {
      "id": 62,
      "providerId": "CD_S2024014",
      "name": "2024 Toyota AFL Premiership",
      "shortName": "Premiership",
      "currentRoundNumber": 28
    },
    {
      "id": 52,
      "providerId": "CD_S2023014",
      "name": "2023 Toyota AFL Premiership",
      "shortName": "Premiership",
      "currentRoundNumber": 28
    }
  ]
}

TEST_ROUND_API_RESPONSE = {
  "meta": {
    "code": 200,
    "pagination": {
      "page": 0,
      "numPages": 3,
      "pageSize": 10,
      "numEntries": 25
    }
  },
  "rounds": [
    {
      "id": 1146,
      "providerId": "CD_R202501400",
      "abbreviation": "OR",
      "name": "Opening Round",
      "roundNumber": 0,
      "byes": [
        {
          "id": 1,
          "providerId": "CD_T10",
          "name": "Adelaide Crows",
          "abbreviation": "ADEL",
          "nickname": "Crows",
          "club": {
            "id": 3,
            "providerId": "CD_O1",
            "name": "Adelaide Crows",
            "abbreviation": "Crows",
            "nickname": "Crows"
          },
          "teamType": "MEN"
        }
      ]
    }
  ]
}

@patch("afl_api._fetch_season_info")
def test_fetch_season_id(fetch_season_info_mock):
    fetch_season_info_mock.return_value = TEST_COMPSEASON_API_RESPONSE
    result = fetch_season_id(2025)
    assert result == 73

@patch("afl_api._fetch_round_info")
def test_fetch_round_id(fetch_round_info_mock):
    fetch_round_info_mock.return_value = TEST_ROUND_API_RESPONSE
    result = fetch_round_id(0, 2025)
    assert result == 1146

