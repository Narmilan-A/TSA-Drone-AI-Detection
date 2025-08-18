# Using QGIS for Annotation

## Learning Objectives
- Load drone imagery into QGIS.
- Create and edit vector layers for vegetation annotation.
- Save and export georeferenced annotation files.

---

## Step 1 – Open the Imagery
1. Launch QGIS.
2. Drag the `.tif` orthomosaic into the Layers panel.

---

## Step 2 – Create an Annotation Layer
1. Go to **Layer → Create Layer → New Shapefile Layer**.
2. Select **Polygon** geometry and **apply coordinate system (CRS)**.
3. Add the following fields:
   - `class_id` (integer)
   - `class_name` (text)
   - `confidence_level` (text) — e.g., 25%, 50%, 75%, 100%
   - `remarks` (text) — optional notes or comments
4. Save the shapefile in your `annotations` folder (**with suitable filename**).
---

## Step 3 – Start Annotating
Before starting annotation, enable the required toolbars for easier digitizing:

1. Right-click anywhere on the toolbar area at the top of QGIS.
2. In the context menu, enable the following toolbars by checking them:
   - **Advanced Digitizing Toolbar**
   - **Digitizing Toolbar**
   - **Shape Digitizing Toolbar**

Once toolbars are enabled, proceed with annotation:

3. Select your polygon vector layer in the Layers panel.
4. Click the **Toggle Editing** button (pencil icon) to enable editing mode.
5. Click **Add Polygon Feature** (usually a polygon icon).
6. Enable **Stream Digitizing Mode**:
   - Click the **Toggle Stream Digitizing Mode** button (it looks like a pencil drawing a curved line).
7. Now, draw the polygon by clicking or dragging on the map canvas:
   - In stream digitizing mode, you can draw more fluidly by moving the mouse cursor.
8. After completing the polygon, right-click to finish drawing.
9. Enter the attribute values (`class_id`, `class_name`, `confidence_level`, `remarks`) in the pop-up form.
10. Repeat for all vegetation patches to annotate.

---

## Step 4 – Save Your Work Frequently
- Save the shapefile in your `annotations` folder.

---

## Step 5 – Colour Coding (Optional but Recommended)

1. Right-click the annotation layer in the Layers panel and select **Properties**.
2. In the Layer Properties window, go to the **Symbology** tab.
3. From the drop-down menu at the top, select **Categorized**.
4. For the **Value** field, choose `class_id` or `class_name` (whichever you prefer to use for coloring).
5. Click the **Classify** button to automatically generate a list of unique classes with default colors.
6. Review the colors assigned to each class:
   - To change a color, click the color square next to the class name.
   - Pick a new color that matches the standard colour codes from your Annotation Guidelines.
7. After adjusting colors as needed, click **Apply** to preview changes on the map.
8. If satisfied, click **OK** to close the Layer Properties window.

This coloring improves visualization and helps easily distinguish vegetation classes while annotating.

---
