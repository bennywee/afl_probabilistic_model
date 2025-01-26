import requests

#TODO: move to consts file
COMPETITION_ID = 1

def fetch_season_info():
    result = requests.get(f"https://aflapi.afl.com.au/afl/v2/competitions/{COMPETITION_ID}/compseasons?pageSize=100")
#    import pdb; pdb.set_trace()
    return result.json()

def parse_json():
    raise NotImplementedError

if __name__ == "__main__":
    result = fetch_season_info()
    print(result)
    
