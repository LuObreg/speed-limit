# Exercise 06: Data Cleaning and Imputation

## Assignment

Inspect the SpeedLimit data. What cleaning is needed? Where should imputation
be used, and how?

## Existing Evidence

The supplied data contains sensor types, sensor mappings, timestamped sensor
readings, accident and near-accident records, and speed-limit readings stored
as sensor ID `-1`. The application performs the following cleaning:

- Parse Year N sensor readings from a semicolon-delimited file with decimal
 commas. Year N+1 is headerless, so its seven-column order is assumed to
 match Year N.
- Discard licence-plate values at ingestion; retain only required accident
 event fields.
- Normalize configured Fahrenheit temperature readings to Celsius and lux
 illuminance to millilux.
- Map configured sensors to weather/light fields; keep sensor 19 out because
 it has readings but no sensor-map entry. Sensor 16 is assumed to be traffic
 and is not part of this model's features.
- Reject measurements outside configured physical bounds and drop rows whose
 time or sensor ID cannot be parsed. Floor timestamps to the hour, keep the
 last reading per sensor and hour, and average the sensors of each
 measurement type.
- Interpolate only a complete internal gap of at most six hours. Leave longer
 gaps wholly unfilled; do not partly fill their edges. Live inference does
 not impute missing readings.
- Count configured near-accident event types in the event hour and join those
 counts to hourly sensor features. Keep accident labels separate from model
 inputs.

Issues and limitations to preserve in the analysis:

- The sensor-type code `T` is used for both temperature and traffic, and the
 per-sensor mapping and units are not confirmed by the data owner.
- The Year N+1 column order is an assumption because the file has no header.
- Reported gaps, outliers, and duplicate records should be re-profiled against
 the exact source rows. Conflicting duplicates should not be resolved by
 “last value wins” without data-owner approval.
- Wind direction is circular: 0 and 360 degrees are equivalent; arithmetic
 averaging can be wrong. The zero-wind handling is not settled.
- Sensor averaging and the six-hour interpolation limit need scientific
 justification for safety-related use.
- The speed for an hour is the last speed reading in that hour, while events
 are grouped by event hour. The actual speed at event time may be unknown.
- Data covers January periods only and cannot establish year-round
 representativeness.

Imputation should be restricted to short, fully bounded internal gaps where
interpolation is meaningful for that measurement. Do not invent labels,
extrapolate across long gaps, or silently treat missing values as safe
conditions. Sensor and data-owner confirmation is needed before treating the
current choices as authoritative.
