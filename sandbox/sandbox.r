library(fitzRoy)
df <- fetch_results(season = 2020, round_number = 1, comp = "AFLM", source = "afltables")

cols <- c(
    "match.homeTeam.name",
    "match.awayTeam.name",
    "round.year",
    "round.roundNumber",
    "homeTeamScore.matchScore.totalScore",
    "homeTeamScore.matchScore.goals",
    "homeTeamScore.matchScore.behinds",
    "homeTeamScore.rushedBehinds",
    "homeTeamScore.minutesInFront",
    "awayTeamScore.matchScore.totalScore",
    "awayTeamScore.matchScore.goals",
    "awayTeamScore.matchScore.behinds",
    "awayTeamScore.rushedBehinds",
    "awayTeamScore.minutesInFront"
)

df[cols] |> View()

fetch_fixture_footywire(
  season = 2025
)

reward <- function(p) {
  1+log2(1+p)
}

cost <- function(p) {
  1+log2(1-p)
}

x = seq(0.5,1, 0.001)

library(ggplot2)
ggplot() +
  geom_line(aes(x=x, y= reward(x)))

library(dplyr)
library(ggplot2)
library(fitzRoy)
t = fetch_player_stats_afltables(season = c(2020:2025))
df = t |> filter(First.name == "Sam", Surname == "Draper") 

df |> filter(Season >=2023) |>
  ggplot(aes(x = as.factor(Date), y = Goals)) +
  geom_bar(stat='identity') +
  theme_minimal() +
  labs(x = "",
       title = "Sam Draper Goals since 2023") +
  theme(axis.title.x=element_blank(),
        axis.text.x=element_blank())


