
fintech_appendices.xlsx — the full dataset for 200 Scottish fintech firms, in four sheets: firm profile data extracted from the FinTech Scotland directory (Appendix B), the investment-readiness scoring output with a score and justification for each of the six Payne scorecard dimensions (Appendix C), the bank relationship classification (Appendix D), and the FCA Regulatory Sandbox matching (Appendix E)

fintech_extraction.ipynb — the notebook used to extract company profile data from the FinTech Scotland directory, producing the firm name, profile URL, funding stage, years trading, employee count, sector, valuation, overview and story fields.

score_comps.py — the scoring script, which submits each firm's profile text to the model together with the standardised rubric, parses the returned dimension scores and justifications, applies the Payne weights and calculates the composite score.
