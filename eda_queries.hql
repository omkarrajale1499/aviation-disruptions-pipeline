USE flight_delay;

-- Q1: Annual delay statistics
SELECT year,
       COUNT(*) as flights,
       ROUND(AVG(dep_delay), 2) as avg_delay,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed,
       ROUND(PERCENTILE_APPROX(dep_delay, 0.95), 2) as p95_delay
FROM all_flights
GROUP BY year
ORDER BY year;

-- Q2: Top 20 worst airports by delay rate (min 10k flights)
SELECT origin,
       COUNT(*) as flights,
       ROUND(AVG(dep_delay), 2) as avg_delay,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY origin
HAVING COUNT(*) >= 10000
ORDER BY pct_delayed DESC
LIMIT 20;

-- Q3: Delay rate by season and hour-of-day
SELECT season, crs_dep_hour,
       COUNT(*) as flights,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY season, crs_dep_hour
ORDER BY season, crs_dep_hour;

-- Q4: Carrier comparison
SELECT carrier,
       COUNT(*) as flights,
       ROUND(AVG(dep_delay), 2) as avg_delay,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY carrier
ORDER BY pct_delayed DESC;

-- Q5: Weather severity impact
SELECT weather_severity,
       COUNT(*) as flights,
       ROUND(AVG(dep_delay), 2) as avg_delay,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY weather_severity
ORDER BY weather_severity;

-- Q6: Top 20 worst routes
SELECT origin, dest,
       COUNT(*) as flights,
       ROUND(AVG(dep_delay), 2) as avg_delay,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY origin, dest
HAVING COUNT(*) >= 1000
ORDER BY pct_delayed DESC
LIMIT 20;

-- Q7: Monthly trend
SELECT year, month,
       COUNT(*) as flights,
       ROUND(AVG(dep_del15) * 100, 2) as pct_delayed
FROM all_flights
GROUP BY year, month
ORDER BY year, month;