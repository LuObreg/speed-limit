# Exercise 05: Feature Engineering

## Assignment

Define the features for machine learning using the SpeedLimit data and explain
how to compute them.

## Existing Evidence

The model has one training row per hour. Sensor readings are cleaned,
normalized, and aggregated to hourly weather/light measurements. Sensor ID
`-1` records the speed limit in force. Accident events are assigned to their
hour and counted to form the target. Training rows without a valid speed are
excluded.

The nine model features are:

1. `illuminance_millilux`: hourly illuminance converted to millilux.
2. `water_level_um`: hourly water-level measurement in micrometers.
3. `temperature_c`: hourly temperature normalized to degrees Celsius.
4. `speed_limit_kmh`: recorded speed limit for the hour during training; a
   candidate speed is supplied during inference.
5. `darkness_score`: configured darkness threshold minus illuminance,
   normalized by that threshold and clipped to the range 0 to 1.
6. `ice_score`: clipped wetness score multiplied by a clipped freezing/cold
   score.
7. `total_danger_index`: configured weighted sum of darkness and ice scores.
8. `hour_sin`: sine of the hour-of-day angle on a 24-hour cycle.
9. `hour_cos`: cosine of the same angle, so midnight and late-night hours are
   represented as neighboring times.

The target is the number of near-accident events in the same hour, with zero
used when no event was recorded. The near-accident categories are skidding,
close approach to another car, and close approach to the guardrail.

The same feature transformation is used for training and inference. Accident
counts are labels, not inference features. The hourly aggregation cannot
establish which speed was active at the exact time of an event if the limit
changed within that hour. Sensor 16 is assumed to be traffic and is not used;
sensor 19 has no confirmed type and is excluded. Additional wind, traffic,
noise, minute/second, or log-transformed illuminance features require domain
justification and validation before inclusion.
