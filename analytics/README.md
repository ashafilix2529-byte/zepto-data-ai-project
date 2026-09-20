# Analytics Pipeline

Run `python analytics_pipeline.py`.

The notebook/script follows the required sequence: one Titanic load, offline CSV fallback, profiling/cleaning, EDA, exploratory standardization, stratified classification split, train-only preprocessing, three classifiers, imbalance comparison, Random Forest GridSearchCV with OOB, fare regression, and saved end-to-end joblib pipeline.

### Required written interpretations
After running, add the observed numeric results to this README or a notebook Markdown cell:
- missing percentages and threshold decisions
- fare mean/median/mode and skewness
- age/fare IQR outlier counts
- sex, class, and sex+class survival rates
- two strongest correlation pairs
- 2–4 sentence interpretation for each of the four multivariate charts
- standardization before/after means and standard deviations
- stratification rationale
- imbalance comparison conclusion
- regression heteroscedasticity conclusion
- final classifier deployment recommendation based on measured metrics
