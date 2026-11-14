# Exercise 07: Year N+1 Evaluation and Retraining

## Assignment

Analyze the sensor readings for the year after the training year and decide
whether the neural network should be retrained.

## Existing Evidence

The Year N model was evaluated on the following Year N+1 period before the
new period was used for annual retraining. The saved evaluation reports 744
hours, of which 389 triggered the weather/light route. For triggered hours,
the observed mean was 1.031 near-accidents per hour, with a reported 95%
percentile-bootstrap interval of 0.933 to 1.134. The predicted triggered-hour
mean was 1.236. The model MAE (mean absolute error) was 0.874, compared with
0.873 for the mean baseline. The interval resamples individual hours and does not account for
temporal autocorrelation.

These figures do not show that the model beats the baseline, and the observed
mean is slightly above the target of one event per triggered hour. They do
not establish that retraining is necessary or that it will improve the model.
The practical recommendation is to retain the evaluate-first workflow, not
to claim an automatic retraining trigger: first compare the frozen model with
an agreed acceptance and drift criterion, then retrain only when new data and
the criterion justify it. Evaluate the retrained model on a later unseen
period. Year N+1 cannot be both its training data and its independent test.

Because the data is observational, neither this evaluation nor candidate
speed predictions prove that changing a speed limit causes fewer
near-accidents. The recorded figures should be rerun and dated before being
presented as current results.
