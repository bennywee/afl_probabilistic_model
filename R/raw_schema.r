ladder_schema <- arrow::schema(
    Season = arrow::int32(),
    Team = arrow::string(),
    Round.Number = arrow::int32(),
    Season.Points = arrow::int32(),
    Score.For = arrow::int32(),
    Score.Against = arrow::int32(),
    Percentage = arrow::float64(),
    Ladder.Position = arrow::int32()
)

results_schema <- arrow::schema(
    Game = arrow::int32(),
    Date = arrow::date32(),
    Round = arrow::string(),
    Home.Team = arrow::string(),
    Home.Goals = arrow::int32(),
    Home.Behinds = arrow::int32(),
    Home.Points = arrow::int32(),
    Away.Team = arrow::string(),
    Away.Goals = arrow::int32(),
    Away.Behinds = arrow::int32(),
    Away.Points = arrow::int32(),
    Venue = arrow::string(),
    Margin = arrow::int32(),
    Season = arrow::int32(),
    Round.Type = arrow::string(),
    Round.Number = arrow::int32()
)
