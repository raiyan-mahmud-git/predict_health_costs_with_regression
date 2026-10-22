# Dataset attribution

The example data is the public FreeCodeCamp health-costs challenge dataset,
served from <https://cdn.freecodecamp.org/project-data/health-costs/insurance.csv>.
The attached challenge notebook uses this same URL. The source referenced here
does not state a separate dataset license; this repository does not claim
ownership of or assign a license to the dataset. Review the source project's
terms before redistributing the CSV.

The model and its outputs are for education and experimentation. This is a
small public teaching dataset, not a current insurance rate table or a basis
for real pricing, underwriting, or individual financial decisions.

## AHRQ Medical Expenditure Panel Survey (MEPS)

The separate annual health-care expenditure estimator uses the **2024 Full
Year Consolidated Public Use File (HC-256)** from the Agency for Healthcare
Research and Quality's Medical Expenditure Panel Survey:

- Data file and file formats: <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-256&prfricon=yes>
- Documentation: <https://meps.ahrq.gov/data_stats/download_data/pufs/h256/h256doc.shtml>
- Codebook: <https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_codebook.jsp?PUFId=H256>

The selected target is `TOTEXP24` (total health-care expenditures in 2024),
with adult BMI (`ADBMI42`), age (`AGE24X`), self-rated health (`RTHLTH53`),
sex (`SEX`), region (`REGION24`), and cigarette smoking frequency (`OFTSMK53`)
used as model inputs. The model
uses the PSAQ/ESAQ expenditure person weight (`PESAQWT24F`) and retains adults
with valid feature, outcome, and weight values. Household groups are kept
together during validation. The MEPS result is an estimate of total annual
health-care expenditure; it is **not** an insurance premium or quote and is
not comparable to the FCC dataset's target.

AHRQ's MEPS documentation includes a statutory data-use agreement: the data
may be used for statistical reporting and analysis, and attempts to identify
individuals or link records outside approved MEPS/NHIS arrangements are
prohibited. The raw data download and derived local model files are ignored
by Git and should not be committed. Cite AHRQ and MEPS when reporting results.
