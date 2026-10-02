-- FraudLens portfolio analytics
SELECT Class AS is_fraud, COUNT(*) AS transactions,
       ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 4) AS share_pct
FROM transactions GROUP BY Class;

SELECT Class AS is_fraud,
       ROUND(AVG(Amount),2) AS avg_amount,
       ROUND(MIN(Amount),2) AS min_amount,
       ROUND(MAX(Amount),2) AS max_amount
FROM transactions GROUP BY Class;

SELECT CAST((Time / 3600) % 24 AS INTEGER) AS hour_of_day,
       COUNT(*) AS transactions,
       SUM(Class) AS frauds,
       ROUND(100.0 * AVG(Class),4) AS fraud_rate_pct
FROM transactions GROUP BY hour_of_day ORDER BY hour_of_day;
