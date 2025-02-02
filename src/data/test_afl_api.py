from unittest.mock import patch
# TODO: fix import *
from afl_api import *

TEST_RESPONSE = {
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

@patch("afl_api._fetch_season_info")
def test_fetch_season_id(fetch_season_info_mock):
    fetch_season_info_mock.return_value = TEST_RESPONSE
    result = fetch_season_id(2025)
    assert result == 73

