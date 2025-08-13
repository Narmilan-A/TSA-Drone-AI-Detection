# Using ArcGIS Pro for Annotation

## Learning Objectives
- Load drone imagery into ArcGIS Pro.
- Create feature classes for annotations.
- Export shapefile.

---

## Method 1: ArcGIS Pro Image Analyst Extension

The first method utilises the advanced capabilities of ArcGIS Pro (v3.1 or higher), specifically the **Image Analyst** extension. This extension allows for a streamlined, semi-automated image labelling process via tools such as the **Training Samples Manager**, which is optimised for supervised classification workflows.

- This method is efficient and significantly reduces labelling time.
- Requires the **Image Analyst** license.
- Ideal for users with access to licensed advanced GIS tools.

---

### Step 1 – Verify Image Analyst License and Enable Extension
1. Open ArcGIS Pro.
2. Go to **Project > Licensing**.
3. Confirm that the **Image Analyst** extension is licensed and enabled.
4. If not enabled, activate it via your organization’s license manager.

---

### Step 2 – Load Orthomosaic Imagery
1. Start a new ArcGIS Pro project.
2. Add the drone orthomosaic (e.g., `.tif` file) to your map.

---

### Step 3 – Open Training Samples Manager
1. Go to the **Imagery** tab on the ribbon.
2. Click on **Classification Tools > Training Samples Manager** to open the window.
3. This tool allows you to create, edit, and manage labelled samples efficiently.

---

### Step 4 – Create a New Training Sample
1. Click **Create New Schema** in the Training Samples Manager.
2. Right-click the new schema and select **Edit Properties**.
3. Rename the schema to a meaningful name and Save it.
4. Click **Save Current Classification Schema**, select the output location, and save the schema file (xxxxx.ecs).
5. Right-click the created schema again and select **Add New Class**.
6. In the dialog:
   - Enter the **Name** (e.g., `class_name`).
   - Assign the corresponding **Value** (e.g., `class_id` integer).
   - Click the color box to select a suitable color for this class.
   - Click **OK** to add the class.
7. Repeat step 6 for all classes you need to label.
8. Click a class and Choose **Freehand** as the sample shape for digitising polygons.
9. Use the mouse **left-click and drag** to draw polygons around vegetation patches, then release the left-click to finish each polygon.
10. Repeat step 8 and 9 for other classes.

---

### Step 5 – Export / Save Training Samples
1. When labelling is complete, export / save the training samples as a shapefile or feature class.
2. These labelled polygons can then be used as ground truth data for machine learning.

---

### Tips:
- Regularly save your project and training samples.
- Consult the Image Analyst help documentation for advanced labelling and classification workflows.

---

![Screenshot of Licensing details and Training Samples Manager](https://github.com/user-attachments/assets/ca29eba1-62c0-4fe0-bc14-ea90f9e996b5)  
*Figure 1: Screenshot of Licensing details and Training Samples Manager*

---
### Note on Adding Attributes to Training Samples in ArcGIS Pro

When using the **Training Sample Manager** under the **Imagery Classification** tools in ArcGIS Pro:

- You can create classes and label your training samples.
- The training samples are saved as a **feature class** with default attributes:  
  - **Class Name**  
  - **Value** (class ID)

**Important:**  
You **cannot add new attribute fields directly within the Training Sample Manager or Classification tool.**

To add additional attribute fields (for example, metadata like "Collector", "Date", or "Confidence"), you need to:

1. Locate the training samples feature class saved on your disk or in your geodatabase.
2. Add the training samples feature class to your map in ArcGIS Pro.
3. Open the **Fields view** or **attribute table design** for the training samples layer.
4. Add new fields with your desired names and data types.
5. Save the changes and then edit the attribute table to populate your new fields.

This process allows you to enrich your training samples with additional metadata outside of the classification interface.

---
## Method 2: ArcGIS Pro Feature Class Tool

The second approach uses the **Create Feature Class** tool available in ArcGIS Pro. This method is more manual but does not require any additional licensing beyond a standard ArcGIS Pro installation.

- Suitable for those without access to the Image Analyst extension.
- Requires more time for manual labelling and polygon digitisation.
- Produces labelled vector files (e.g., shapefiles or feature classes) suitable for downstream machine learning model development or analysis.
---
### Step 1
- Click **View** → **Geoprocessing** to open the Geoprocessing pane.
---
### Step 2
- In the Geoprocessing search bar, type **Create Feature Class** and click on it from the list.
---
### Step 3
In the **Create Feature Class** tool:
- Set **Location** to your desired geodatabase.
- Enter a **Name** for the new feature class.
- Set **Geometry Type** to *Polygon*.
- Set **Coordinate Reference System (CRS)** to match your UAV imagery.
Click **Run**.
---
### Step 4
- After running, check the **Contents** pane — you should see the new feature class added as a vector layer.
---
### Step 5
- Select the new feature class layer in the **Contents** pane.
- Click the **Edit** tab → **Create** to open the **Create Features** panel.
---
### Step 6
- In the **Create Features** panel, choose your polygon template, then click the **Freehand** tool.
---
### Step 7
- Use the **Freehand** tool to draw a polygon around the area corresponding to a specific class (e.g., vegetation type from GNSS RTK point data).
---
### Step 8
- Complete the polygon by releasing the mouse click. The polygon will be added to your feature class.
---
### Step 9
- Repeat the process for each region of interest, based on the available GNSS RTK points for each class.
---
### Step 10
To assign labels:
- Right-click the feature class in the **Contents** pane.
- Select **Attribute Table**.
- Edit the relevant field to input the class name for each polygon.
---
### Step 11
Apply symbology for better visualisation:
- Right-click the feature class → **Symbology**.
- Choose **Categorized**.
- Click **Classify** and adjust colours for each class if needed.
- Click **OK**.
---
### Step 12
- Save the edits by clicking **Save Edits** in the **Edit** tab, and ensure all polygons are stored in your project geodatabase.
---
![Processing steps for labelling using feature class tool - Part 1](https://github.com/user-attachments/assets/1678f752-872a-4408-b59f-376a6f9fe37e)  
*Figure 2: Processing steps for labelling using feature class tool (Part-1)*

![Processing steps for labelling using feature class tool - Part 2](https://github.com/user-attachments/assets/e680df51-8ffb-48a7-8edd-42b228008b97)  
***Figure 3: Processing steps for labelling using feature class tool (Part-2)*
**---

## Summary

Both methods provide viable approaches to image labelling within ArcGIS Pro, depending on the tools and licensing available:

- **Image Analyst Extension**: Efficient, advanced functionality; ideal for users with access to licensed tools.
- **Feature Class Tool**: Manual but accessible; suitable for broader use cases with no additional licensing cost.

---
