import requests

#TODO: move to consts file
COMPETITION_ID = 1

def _fetch_season_info():
    result = requests.get(f"https://aflapi.afl.com.au/afl/v2/competitions/{COMPETITION_ID}/compseasons?pageSize=100")
    return result.json()

def _parse_json(season_info):
    print(season_info)
    raise NotImplementedError

def fetch_season_id(year):
    season_info = _fetch_season_info()
    _parse_json(season_info)
    raise NotImplementedError

if __name__ == "__main__":
    result = _fetch_season_info()
    print(result)
    
