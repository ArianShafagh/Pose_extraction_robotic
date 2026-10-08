# Progress report

`docs/moppose_progress_report.pdf` is generated from the project data:

```bash
uv run python docs/report/make_figures.py                     # figures -> docs/report/fig/
uv run --with reportlab python docs/report/build_pdf.py       # PDF -> docs/moppose_progress_report.pdf
```

Model survey: `uv run --with reportlab python docs/report/build_models_pdf.py` -> `docs/landmark_models_survey.pdf`
