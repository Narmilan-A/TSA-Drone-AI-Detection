# Ground Truth Data and Region of Interest (ROI) Development

## Overview
This document explains how to use the GNSS RTK (Real-Time Kinematic) ground truth points collected for different vegetation classes in the vegetation mapping project. These accurate location points provide the foundation for creating Regions of Interest (ROIs), which guide manual annotation of UAV imagery.

---

## Step 1 – Understanding GNSS RTK Ground Truth Points
- GNSS RTK points provide precise spatial locations (centimeter-level accuracy) of sample vegetation types such as A, B, C, and D species.
- Each point is tagged with:
  - Class ID and name
  - Timestamp and metadata
  - Geographic coordinates (latitude, longitude, elevation)
- These points serve as verified “ground truth” for training and validating machine learning models.

---

## Step 2 – Access and Review Ground Truth Data
- Ground truth points are provided as a georeferenced CSV or shapefile.
- Load the data into GIS software such as QGIS or ArcGIS Pro:
  1. Open QGIS or ArcGIS Pro.
  2. Import the ground truth file.
  3. Inspect the distribution of points across the sites.
- Confirm that the points correspond to visible features on the drone orthomosaics.

---

## Step 3 – Develop Regions of Interest (ROIs)
- Using the GNSS points as anchors, create polygon ROIs around each vegetation sample cluster:
  1. Buffer the points by an appropriate radius to form square polygons representing sampling areas (Prefer).
  2. Alternatively, manually digitize polygons guided by the point locations and visible vegetation boundaries in the imagery.
- Save ROIs in a separate layer for annotation guidance.

---

## Step 4 – Use ROIs to Guide Manual Annotation
- The ROIs represent spatial areas where vegetation types are confirmed.
- Annotators should use these polygons to:
  - Focus their manual labelling within the ROI boundaries.
  - Ensure labels match the ground truth classes linked to each ROI.
- This approach helps improve annotation consistency and accuracy.

---

## Step 5 – Quality Control and Iteration
- After initial ROI creation and annotation, review results for spatial consistency.
- Adjust polygon boundaries if necessary based on drone imagery and annotator feedback.
- Iterate to refine ROIs for best representation of vegetation classes.

---

## Tips
- Always cross-check GNSS point metadata before creating ROIs.
- Use transparent polygon layers in GIS to visualise ROIs over orthomosaics.

---

## Summary
Using GNSS RTK ground truth points to develop ROIs ensures your manual annotation is spatially accurate and scientifically robust. This methodology supports high-quality training datasets for machine learning-based vegetation mapping.

---
