import requests

#TODO: move to consts file
COMPETITION_ID = 1

def _fetch_season_info():
    result = requests.get(f"https://aflapi.afl.com.au/afl/v2/competitions/{COMPETITION_ID}/compseasons?pageSize=100")
    return result.json()

def fetch_season_id(year):
    season_info = _fetch_season_info()
    # this filter logic is brittle, but it matches fitzRoy https://github.com/jimmyday12/fitzRoy/blob/006ebef7bc7d892aa8b999278ea460898c19200d/R/helpers-afl.R#L251
    matching_comp_seasons = [compSeason for compSeason in season_info["compSeasons"]
                    if str(year) in compSeason["name"]]

    # TODO: validate that exactly one result here?
    return matching_comp_seasons[0]['id']

def _fetch_round_info(round_number, season_id):
    url = f"https://aflapi.afl.com.au/afl/v2/compseasons/{season_id}/rounds"
    result = requests.get(url)
    return result.json()

# Implementing this function, sans future functionality
# https://github.com/jimmyday12/fitzRoy/blob/006ebef7bc7d892aa8b999278ea460898c19200d/R/helpers-afl.R#L271
def fetch_round_id(round_number, year):
    season_id = fetch_season_id(year)
    round_info = _fetch_round_info(round_number, season_id)
    matching_rounds = [round_ for round_ in round_info["rounds"]
                       if round_['roundNumber'] == round_number]
    return matching_rounds[0]['id']


if __name__ == "__main__":
    import pprint
    #result = _fetch_season_info()
    result = _fetch_round_info(0, 73)
    pprint.pp(result)
    
