# RentLens — Real Data Quality Report

Source: MagicBricks (Powai / Mulund / Andheri East rental search results)
Raw rows scraped: 621
Final clean rows: 573  (92.3% retained)

## Pipeline steps

| Step | Rows before | Rows after | Dropped | Note |
|---|---|---|---|---|
| drop_missing_rent | 621 | 609 | 12 | monthly_rent is the target variable; can't model without it |
| resolve_bhk | 609 | 608 | 1 | Studio -> 1 BHK (industry convention); rest had no BHK info to impute from |
| reconcile_area | 608 | 607 | 1 | super-area -> carpet via x0.7 loading factor for 142 rows (flagged in carpet_area_is_estimated); dropped rows with no area at all |
| drop_missing_furnishing | 607 | 605 | 2 | no furnishing data on the card at all; nothing to impute from |
| rent_outliers | 605 | 593 | 12 | dropped rent < Rs 15,000 (implausible for a full flat in these localities) or > Rs 1000/sqft/month (likely sale-price leakage, not rent) |
| locality_bucketing | 593 | 590 | 3 | assigned to nearest of 3 target-locality centroids by haversine distance; dropped rows > 4,000 m from any centroid (too ambiguous to bucket) |
| dedupe_relistings | 590 | 573 | 17 | dropped exact-match relistings on ['latitude', 'longitude', 'bhk', 'bathrooms', 'carpet_area_sqft', 'monthly_rent'] (same building/spec/rent = same physical unit) |
| building_age_years | 573 | 573 | 0 | only 5.4% of rows have genuine age-bucket text (rest say move-in availability, e.g. 'Immediately' — NOT age); left as NaN rather than guessed |
