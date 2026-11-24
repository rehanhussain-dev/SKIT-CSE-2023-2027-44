**PROJECT ABSTRACT \[Form-1\] SESSION (2026-27)**

**• PROJECT ID:** SKIT/CSE/2023-2027/..

**• BRANCH:** CSE

**• SECTION:** C

**• TITLE OF PROJECT:** Farm Yield Prediction using Very High Spatial Resolution Data

**• SDG MAPPING:** 02 (Zero Hunger)

**• PROJECT TRACK:** R&D / Innovation Projects

**• MANDATORY EXTERNAL EVALUATION OF PROJECT:** Research Paper Publication

**PROBLEM STATEMENT**

In Rajasthan, 75–86% of farmers operate on fields with sizes between 2–5 hectares. Standard 10m satellite pixels blend crops with non-crop boundaries, introducing severe spectral noise. Furthermore, macro-scale models miss localized heat spikes and soil salinity, while standard machine learning models lack biological accountability.

**TECHNOLOGY STACK**

| **Name of Tool / Technology** | **Frontend / Backend** | **Software / Hardware** | **Purpose of Use**                          |
| ----------------------------- | ---------------------- | ----------------------- | ------------------------------------------- |
| Leaflet.js / OpenStreetMap    | Frontend               | Software                | Interactive GIS field polygon selection     |
| Django REST Framework         | Backend                | Software                | Web application framework & API routing     |
| PostgreSQL / PostGIS          | Backend                | Software                | Spatial database storage                    |
| PyTorch (SRGAN)               | Backend                | Software                | Deep learning spatial upscaling             |
| SHAP                          | Backend                | Software                | Explainable AI & stress attribution         |
| PCSE / WOFOST & LSTM          | Backend                | Software                | Biophysical crop engine & temporal modeling |

**PROPOSED PROJECT SPRINTS**

| **S.No.** | **Sprint Name**                     | **Proposed User Story of Respective Sprint**                                                                                                                                                                                                              |
| --------- | ----------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 1.        | Data Assimilation                   | As a data engineer, I want to deploy the spatial database and backend, ingest cadastral field boundaries, and build a clean, validated pipeline for satellite and crop parameter data, so that the modeling pipeline has reliable, analysis-ready inputs. |
| 2.        | Spatial Super-Resolution            | As a machine learning engineer, I want to train and optimize a SRGAN-based super-resolution model that upscales 10m satellite bands to 2.5–3m grids, so that field-level detail is preserved for accurate small-farm predictions.                         |
| 3.        | WOFOST Integration & Physics Fusion | As a machine learning engineer, I want to fuse the PCSE-WOFOST biophysical model with an LSTM-based temporal network using a physics-informed (PINN) loss, so that yield predictions stay biologically accurate and respect crop growth limits.           |
| 4.        | Full Stack Integration              | As a full-stack developer, I want to build an interactive GIS dashboard that masks, aggregates, and visualizes model outputs as zonal yield heatmaps, so that farmers and stakeholders get clear, actionable field-level insights.                        |

**TEAM MEMBER DETAILS**

| **Student Role**             | **Expertise Area**              | **Role in Project**      |
| ---------------------------- | ------------------------------- | ------------------------ |
| Rehan Hussain                | Data Science and Engineering    | Data Engineering         |
| Prashant Khansili            | Deep Learning / Computer Vision | ML Engineering / Testing |
| Pratik                       | Machine Learning / Full Stack   | ML Engineering           |
| Rajput Devesh Bijendra Singh | Full Stack Development          | Full Stack / Testing     |

**Verified & Approved By:  
Mentor's Signature: \_**\_**\_**\_**\_**\_**\____**

**ROLES & RESPONSIBILITY OF TEAM MEMBERS \[Form – 2\]**

**• PROJECT ID:** SKIT/CSE/2023-2027/..

**• BRANCH:** CSE

**• SECTION:** C

**• TITLE OF PROJECT:** Farm Yield Prediction using Very High Spatial Resolution Data

**TEAM LEAD**

**Name: Rehan Hussain SPRINT(s) NAME: Data Assimilation**

| **User Story**                         | **Start Date** | **End Date** | **Details of Task Completed Under the User Story**                                                                                                                                                                |
| -------------------------------------- | -------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Spatial Database & Boundary Ingestion  | 10/08/2026     | 20/09/2026   | Deployed the PostGIS spatial database extensions, stood up the Django REST framework backend structure, and vectorized/stored field boundaries into PostGIS spatial tables.                                       |
| GEE Ingestion & Patch Export           | 21/09/2026     | 01/11/2026   | Wrote GEE scripts for Sentinel-2C/Landsat-9 with cloud/shadow masking; computed NDVI/NDRE/SWIR-SI matrices and automated 128×128 GeoTIFF patch export.                                                            |
| Training Data QA                       | 02/11/2026     | 13/12/2026   | Audited exported patch batches for artefacts/gaps in the training set and re-exported as needed to ensure clean inputs.                                                                                           |
| Crop Parameter Validation & API Schema | 14/12/2026     | 24/01/2027   | Extracted and validated ICAR-DRMR / GCES 2026 crop parameter records; designed the JSON payload contract between model outputs and the REST API.                                                                  |
| Sprint Sync & Pipeline Maintenance     | 25/01/2027     | 30/03/2027   | Ran bi-weekly sprint syncs across all tracks; integrated field validation results back into the spatial data pipeline, resolved discrepancies flagged during benchmarking, and maintained pipeline documentation. |

**MEMBER 1**

**Name: Prashant Khansili SPRINT(s) NAME: Spatial Super-Resolution**

| **User Story**                          | **Start Date** | **End Date** | **Details of Task Completed Under the User Story**                                                                                                                                                                                              |
| --------------------------------------- | -------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Patch Review Pipeline                   | 10/08/2026     | 20/09/2026   | Built the GeoTIFF patch review pipeline; reviewed exported patches for training suitability and flagged boundary/cloud artefacts.                                                                                                               |
| SRGAN Initialization & Fine-Tuning      | 21/09/2026     | 01/11/2026   | Established 4× matrix reconstruction layers on the SRGAN backbone; fine-tuned network weights on local open-source agricultural image patches.                                                                                                  |
| Sub-Pixel Upscaling & Fusion Validation | 02/11/2026     | 13/12/2026   | Upscaled 10m bands to 2.5–3m grids; verified boundary sharpening and pixel-mixing elimination; validated that upscaled bands feeding the temporal LSTM branch are spatially consistent, troubleshooting artefacts as they surfaced in training. |
| Inference Optimization                  | 14/12/2026     | 24/01/2027   | Optimized SRGAN inference latency and throughput for near-real-time dashboard serving.                                                                                                                                                          |
| Field Validation & Documentation        | 25/01/2027     | 30/03/2027   | Compared SR outputs against ground/secondary data across the 3–5 pilot regions; regression-tested the SR pipeline end-to-end and wrote up module documentation.                                                                                 |

**MEMBER 2**

**Name: Pratik SPRINT(s) NAME: WOFOST Integration & Physics Fusion**

| **User Story**                             | **Start Date** | **End Date** | **Details of Task Completed Under the User Story**                                                                                                                                                                                                                               |
| ------------------------------------------ | -------------- | ------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| WOFOST Engine Setup                        | 10/08/2026     | 20/09/2026   | Configured the PCSE-WOFOST Python framework and loaded regional crop parameter profiles.                                                                                                                                                                                         |
| Weather API & LSTM Skeleton                | 21/09/2026     | 01/11/2026   | Connected the NASA POWER weather API (free, publicly accessible — no licensing cost); built the LSTM branch architecture.                                                                                                                                                        |
| LSTM Growth Trajectory Training            | 02/11/2026     | 13/12/2026   | Wired the LSTM branch to ingest the 2.5m upscaled vegetation-index time series from the super-resolution output; trained LSTM modules on time-series vegetation trends across crop stages.                                                                                       |
| PINN Loss & Joint Model Training           | 14/12/2026     | 24/01/2027   | Programmed the Physics-Informed (PINN) loss balancing empirical and biological limits; trained the joint LSTM-WOFOST model to generate localized kg/pixel yield predictions.                                                                                                     |
| Threshold Testing & Explainability Support | 25/01/2027     | 30/03/2027   | Tested predictions against regional crop physiological growth thresholds; tuned model inference for dashboard latency; fed live model outputs into the SHAP explainability node; supported accuracy benchmarking and refined the fusion model against field validation findings. |

**MEMBER 3**

**Name: Rajput Devesh SPRINT(s) NAME: Dashboarding, Masking & System Integration**

| **User Story**                            | **Start Date** | **End Date** | **Details of Task Completed Under the User Story**                                                                                                                                                                                                                            |
| ----------------------------------------- | -------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Frontend Shell & Polygon Tool             | 10/08/2026     | 20/09/2026   | Built the Leaflet.js/OpenStreetMap map shell and a polygon-drawing tool prototype (no backend dependency).                                                                                                                                                                    |
| Django REST API Scaffolding               | 21/09/2026     | 01/11/2026   | Stood up Django REST project structure, auth, and placeholder endpoints.                                                                                                                                                                                                      |
| Masking, Aggregation & SHAP Scaffolding   | 02/11/2026     | 13/12/2026   | Wrote GeoPandas/Rasterio Point-in-Polygon scripts against synthetic prediction grids; implemented polygon-clipping to sum field yield in Metric Tons; wired the SHAP explainability node against mock stress outputs.                                                         |
| API Serialization & Dashboard Integration | 14/12/2026     | 24/01/2027   | Built the endpoints that package spatial yield heatmaps and stress reports per the defined payload schema; built out the full Leaflet.js dashboard UI; swapped mock prediction data for validated model outputs and connected UI requests to the real backend asynchronously. |
| Heatmap Rendering & Final Benchmarking    | 25/01/2027     | 30/03/2027   | Displayed sub-meter crop performance zones and multilingual advisory cards; ran multi-farm regional field testing against secondary crop receipt data; compared yield estimation maps under real-world conditions; fixed UI bugs and finalized the codebase for submission.   |

**Verified & Approved By:  
Mentor's Signature: \_**\_**\_**\_**\_**\_**\____**