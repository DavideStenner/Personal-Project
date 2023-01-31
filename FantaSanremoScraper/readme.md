# Fanta Sanremo

The aim of this project is to create the best possible team with the available information/data.

This was done in two steps:

- Scrape all teams from https://app.fantasanremo.com
- Optimal choice of the team by solving the knapsack problem, using the average order of artist choice within the team as the value.

# How it works

### Scrape Main league

```
python scraper.py
```

### Scrape all league

```
python scraper.py --scrape_all_league
```

### Scrape 60% from TicketOne League
```
python scraper.py --league "Campionato Mondiale" --pct_scrape 0.6

```

### get best team
```
python scraper.py --league "Campionato Mondiale"
python get_best.py --league "Campionato Mondiale"
```
### get every best team
```
python scraper.py --scrape_all_league
python get_best.py --get_all
```